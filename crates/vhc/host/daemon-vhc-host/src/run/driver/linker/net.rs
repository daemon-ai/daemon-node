// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `net@2` world: content-addressed payloads by handle (`payload_put`/`payload_get`, §3.4),
//! direct peer streams under credit flow control (§3.3), and `publish` — the signed, sequenced,
//! durable egress door (§6.2/§12).

use wasmtime::{Caller, Linker};

use daemon_vhc_abi::{CHANNEL_DIR_RX_ONLY, NS_NET_V2, PHASE_A_DEFAULT_CHANNEL_TABLE};

use crate::run::driver::host::{build_signed_frame, read_guest, stash, Host};
use crate::run::ops::OpRequest;
use crate::trap::{Trap, TrapCode};

/// REL-10 (reliability spec §12): the by-ref family folds a payload carries WHEN it is the
/// host's §10.2 checkpoint-document shape — the committed run evidence that extends the
/// admitted artifact set for the putting incarnation. Any other payload (including one that
/// happens to be CBOR but not the doc shape) yields nothing.
fn checkpoint_evidence_folds(bytes: &[u8]) -> Vec<[u8; 32]> {
    daemon_vhc_proto::det_state::decode_checkpoint_doc(bytes)
        .map(|(_, sections)| {
            sections
                .iter()
                .filter_map(|s| match s {
                    daemon_vhc_proto::det_state::CkptDocSection::ByRef(_, family) => {
                        Some(family.fold.0)
                    }
                    daemon_vhc_proto::det_state::CkptDocSection::Inline(..) => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Link the `net@2` imports.
#[allow(clippy::too_many_lines)]
pub(super) fn link(linker: &mut Linker<Host>) -> Result<(), wasmtime::Error> {
    // ---- net@2 minor 1: content-addressed payloads by handle (§3.4) — both complete async -------
    linker.func_wrap(
        NS_NET_V2,
        "payload_put",
        |mut c: Caller<'_, Host>, buffer: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("payload_put")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let bytes = st
                    .buffers
                    .resolve(buffer)
                    .map_err(|code| Trap::new(code, "payload_put", None, "buffer handle"))?;
                let request = OpRequest::PayloadPut { bytes };
                let op = st.ops.begin(request.clone()).map_err(|code| {
                    Trap::new(code, "payload_put", None, "max_outstanding grant (§2.3)")
                })?;
                // REL-10 (reliability spec §12): a put whose bytes carry the host's §10.2
                // checkpoint-document shape is this run's own rotation minting COMMITTED run
                // evidence — the document's by-ref family folds join the admitted artifact
                // set (the `da_migrate` precedent, generalized to the continuous case), so a
                // later re-fetch of a rotated fold (e.g. after the sealed store evicted the
                // superseded generation) is served from the content plane instead of trapping
                // GrantViolation. Deterministic guest output (the put CALL, not its service),
                // so replay reproduces the same extension at the same point; a fetch of a
                // hash with NO committed evidence still traps.
                let OpRequest::PayloadPut { bytes: put_bytes } = &request else {
                    unreachable!("constructed above")
                };
                let evidence_folds = checkpoint_evidence_folds(put_bytes);
                st.op_requests.push((op, request));
                st.note_egress();
                drop(st);
                for fold in evidence_folds {
                    c.data_mut().granted_artifacts.insert(fold);
                }
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "payload_get",
        |mut c: Caller<'_, Host>, hash_ptr: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("payload_get")?;
                let hash_bytes = read_guest(c, hash_ptr, 32)?;
                let hash: [u8; 32] = hash_bytes.as_slice().try_into().expect("32-byte span");
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st
                    .ops
                    .begin(OpRequest::PayloadGet { hash })
                    .map_err(|code| {
                        Trap::new(code, "payload_get", None, "max_outstanding grant (§2.3)")
                    })?;
                st.op_requests.push((op, OpRequest::PayloadGet { hash }));
                st.note_egress();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- net@2 minor 1: direct peer streams under credit flow control (§3.3/§3.4) ---------------
    linker.func_wrap(
        NS_NET_V2,
        "stream_open",
        |mut c: Caller<'_, Host>, peer_ptr: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("stream_open")?;
                let peer_bytes = read_guest(c, peer_ptr, 32)?;
                let peer: [u8; 32] = peer_bytes.as_slice().try_into().expect("32-byte span");
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st
                    .ops
                    .begin(OpRequest::StreamOpen { peer })
                    .map_err(|code| {
                        Trap::new(code, "stream_open", None, "max_outstanding grant (§2.3)")
                    })?;
                st.op_requests.push((op, OpRequest::StreamOpen { peer }));
                st.note_egress();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "stream_accept",
        |mut c: Caller<'_, Host>| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("stream_accept")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st.ops.begin(OpRequest::StreamAccept).map_err(|code| {
                    Trap::new(code, "stream_accept", None, "max_outstanding grant (§2.3)")
                })?;
                st.op_requests.push((op, OpRequest::StreamAccept));
                st.note_egress();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "stream_write",
        |mut c: Caller<'_, Host>, stream: u64, buffer: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("stream_write")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let bytes = st
                    .buffers
                    .resolve(buffer)
                    .map_err(|code| Trap::new(code, "stream_write", None, "buffer handle"))?;
                let op = st
                    .ops
                    .begin(OpRequest::StreamWrite {
                        stream,
                        bytes: bytes.clone(),
                    })
                    .map_err(|code| {
                        Trap::new(code, "stream_write", None, "max_outstanding grant (§2.3)")
                    })?;
                // Credit flow control (§3.3): the transport request is emitted only when the
                // stream's writable credit covers the bytes; otherwise the op is HELD pump-side
                // (still outstanding — the guest's OpId is live) until the receiver's reads
                // replenish credit.
                match st.streams.write(stream, op, bytes.clone()) {
                    Some(true) => {
                        st.op_requests
                            .push((op, OpRequest::StreamWrite { stream, bytes }));
                        st.note_egress();
                    }
                    Some(false) => { /* held for credit */ }
                    None => {
                        st.ops.finish(op);
                        return Err(Trap::new(
                            TrapCode::StaleHandle,
                            "stream_write",
                            None,
                            "unknown or stale stream handle",
                        ));
                    }
                }
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;
    linker.func_wrap(
        NS_NET_V2,
        "stream_read",
        |mut c: Caller<'_, Host>, stream: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("stream_read")?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                if !st.streams.is_live(stream) {
                    return Err(Trap::new(
                        TrapCode::StaleHandle,
                        "stream_read",
                        None,
                        "unknown or stale stream handle",
                    ));
                }
                let op = st
                    .ops
                    .begin(OpRequest::StreamRead { stream })
                    .map_err(|code| {
                        Trap::new(code, "stream_read", None, "max_outstanding grant (§2.3)")
                    })?;
                st.op_requests.push((op, OpRequest::StreamRead { stream }));
                st.note_egress();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- net@2::publish — the signed, sequenced, durable egress door (§6.2/§12) -----------------
    linker.func_wrap(
        NS_NET_V2,
        "publish",
        |mut c: Caller<'_, Host>,
         channel_id: u32,
         payload_ptr: u32,
         payload_len: u32|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("publish")?;
                // The channel table decides class/direction/bounds — never the guest (§6.2).
                let decl = PHASE_A_DEFAULT_CHANNEL_TABLE
                    .iter()
                    .find(|ch| ch.id == channel_id)
                    .ok_or_else(|| {
                        Trap::new(
                            TrapCode::GrantViolation,
                            "publish",
                            None,
                            format!("undeclared channel {channel_id} (§6.2)"),
                        )
                    })?;
                if decl.direction == CHANNEL_DIR_RX_ONLY {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "publish",
                        None,
                        format!("channel {channel_id} is rx-only (§6.2)"),
                    ));
                }
                if payload_len > c.data().max_frame_bytes {
                    return Err(Trap::new(
                        TrapCode::PayloadOverflow,
                        "publish",
                        None,
                        format!(
                            "payload {payload_len} bytes > max_frame_bytes {}",
                            c.data().max_frame_bytes
                        ),
                    ));
                }
                let payload = read_guest(c, payload_ptr, payload_len)?;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                // Atomic commit (§6.2): seq allocation + the tag-4 record + (Phase-A) the spool
                // stand-in are covered by the sink's publish barrier before the guest sees seq.
                let seq = st.sink.next_seq(u64::from(channel_id));
                let frame = build_signed_frame(c.data(), u64::from(channel_id), seq, &payload)?;
                st.sink
                    .publish(u64::from(channel_id), seq, &payload, &frame)
                    .map_err(Trap::from)?;
                st.published.push((u64::from(channel_id), seq, frame));
                st.note_egress();
                // A registered stop cut (§4.4): the run is complete AT this publish — enqueue the
                // Stop in the same critical section, so nothing else can enter the stream first.
                if let Some((n, reason)) = st.stop_cut {
                    if st.published.len() >= n {
                        st.enqueue_stop(reason).map_err(Trap::from)?;
                    }
                }
                Ok(seq)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::checkpoint_evidence_folds;
    use daemon_vhc_proto::det_state::{
        encode_checkpoint_doc, family_fold, CkptDocSection, FamilyRef,
    };
    use daemon_vhc_proto::Hash;

    /// REL-10 (reliability spec §12): ONLY the host's checkpoint-document shape mints evidence
    /// folds — its by-ref families, exactly; inline sections, arbitrary bytes, and non-doc CBOR
    /// yield nothing (a fetch of a hash with no committed evidence keeps trapping).
    #[test]
    fn only_a_checkpoint_document_extends_the_evidence_granted_set() {
        let chunk_hashes = vec![Hash([0x11; 32]), Hash([0x22; 32])];
        let family = FamilyRef {
            fold: family_fold(32, 64, &chunk_hashes),
            byte_len: 64,
            chunk_size: 32,
            chunk_hashes,
        };
        let doc = encode_checkpoint_doc(
            b"manifest",
            &[
                CkptDocSection::Inline("small".into(), vec![1, 2, 3]),
                CkptDocSection::ByRef("master".into(), family.clone()),
            ],
        )
        .expect("doc encodes");

        assert_eq!(
            checkpoint_evidence_folds(&doc),
            vec![family.fold.0],
            "the by-ref folds — and only those — are the committed evidence"
        );
        assert!(
            checkpoint_evidence_folds(b"not a doc").is_empty(),
            "arbitrary bytes mint nothing"
        );
        let cbor_not_doc =
            daemon_vhc_proto::to_canonical_vec(&vec![1u64, 2, 3]).expect("cbor encodes");
        assert!(
            checkpoint_evidence_folds(&cbor_not_doc).is_empty(),
            "CBOR that is not the doc shape mints nothing"
        );
    }
}
