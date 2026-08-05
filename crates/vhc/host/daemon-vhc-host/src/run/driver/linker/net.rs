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
                st.op_requests.push((op, request));
                st.note_egress();
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
