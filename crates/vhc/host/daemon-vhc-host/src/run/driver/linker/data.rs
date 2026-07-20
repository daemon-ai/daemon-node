// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `data@2` world (track B2, architecture §3.2): `fetch` — artifact fetch by committed
//! hash and range, under the admitted-artifact grant and the cumulative read budget — and
//! `register_chunks`, the chunk-addressed corpus registration whose fold must itself be a
//! granted artifact identity.

use std::sync::Arc;

use wasmtime::{Caller, Linker};

use daemon_vhc_abi::{COMP_ERR_GRANT_EXHAUSTED, NS_DATA_V2};

use crate::run::completion::{CompError, CompletionResult, SuccessPayload};
use crate::run::driver::chunks::decode_chunk_descriptor;
use crate::run::driver::host::{read_guest, stash, Host};
use crate::run::driver::pump::PumpState;
use crate::run::ops::OpRequest;
use crate::trap::{Trap, TrapCode};

/// Link the `data@2` imports.
#[allow(clippy::too_many_lines)]
pub(super) fn link(linker: &mut Linker<Host>) -> Result<(), wasmtime::Error> {
    // ---- data@2::fetch — artifact fetch by committed hash + range (track B2; §3.2) --------------
    // The guest names CONTENT, never location: the only inputs are the committed blake3 (edge-
    // pinned in the envelope's artifact map, §5.1) and a byte range — no URL, no locator, no
    // credential crosses this boundary (the resolver + its credentials stay embedder-side).
    // Which artifacts a module may touch is a GRANT: a hash outside the admitted set traps
    // GrantViolation before any op is issued. Completes Ok(BufferHandle) via tag 6 after the
    // pump whole-artifact-verifies + range-slices (see complete_op).
    linker.func_wrap(
        NS_DATA_V2,
        "fetch",
        |mut c: Caller<'_, Host>,
         hash_ptr: u32,
         range_off: u64,
         range_len: u64|
         -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("fetch")?;
                let hash_bytes = read_guest(c, hash_ptr, 32)?;
                let hash: [u8; 32] = hash_bytes.as_slice().try_into().expect("32-byte span");
                let granted = c.data().granted_artifacts.contains(&hash);
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                // [SF-R1] (ABI §12.14): a fold the instance itself sealed is registered — and
                // fetchable — BY CONSTRUCTION: the host built the chunk map as the chunks were
                // emitted, so no `register_chunks` call and no grant entry gate it. Serviced
                // host-locally at the call (the store is in-process; no embedder round-trip),
                // completing through the ordinary Completion protocol so the guest-visible
                // shape is identical to every other fetch.
                if st.state.sealed(&hash).is_some() {
                    let op = service_self_sealed_fetch(&mut st, hash, range_off, range_len)?;
                    st.note_egress();
                    return Ok(op);
                }
                if !granted {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "fetch",
                        None,
                        format!(
                            "artifact {} is not in the admitted artifact set (which artifacts a \
                             module may touch is a grant, architecture §3.2)",
                            hash[..4]
                                .iter()
                                .map(|b| format!("{b:02x}"))
                                .collect::<String>()
                        ),
                    ));
                }
                let chunked = st.chunk_maps.get(&hash).map(|m| (m.byte_len, m.chunk_size));
                let (used, budget) = (st.data_read_used, st.data_read_budget);
                let op = match chunked {
                    // Chunk-addressed (a registered corpus shard): bounds are knowable NOW
                    // (registration pinned the geometry), the read budget is charged at the
                    // call (guest-call-order deterministic), and the embedder is asked for the
                    // chunk-aligned COVERING SPAN only — never the whole shard.
                    Some((byte_len, chunk_size)) => {
                        let end = if range_len == 0 {
                            byte_len
                        } else {
                            range_off.saturating_add(range_len)
                        };
                        if range_off > byte_len || end > byte_len {
                            immediate_fetch_refusal(
                                &mut st,
                                hash,
                                range_off,
                                range_len,
                                daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                                format!(
                                    "range [{range_off}, {end}) out of bounds (registered \
                                     shard is {byte_len} bytes)"
                                ),
                            )?
                        } else {
                            let charge = end - range_off;
                            if budget != 0 && used.saturating_add(charge) > budget {
                                immediate_fetch_refusal(
                                    &mut st,
                                    hash,
                                    range_off,
                                    range_len,
                                    COMP_ERR_GRANT_EXHAUSTED,
                                    format!(
                                        "data-read budget exhausted ({used} of {budget} bytes \
                                         used; {charge} more requested)"
                                    ),
                                )?
                            } else {
                                st.data_read_used += charge;
                                let (span_off, span_len) = daemon_vhc_proto::covering_span(
                                    byte_len, chunk_size, range_off, end,
                                );
                                if span_len == 0 {
                                    // An empty range needs no store round-trip: complete an
                                    // empty buffer at the call (deterministic, journaled).
                                    let request = OpRequest::ArtifactRange {
                                        hash,
                                        range_off,
                                        range_len,
                                        span_off,
                                        span_len,
                                    };
                                    let op = st.ops.begin(request).map_err(|code| {
                                        Trap::new(
                                            code,
                                            "fetch",
                                            None,
                                            "max_outstanding grant (§2.3)",
                                        )
                                    })?;
                                    st.ops.finish(op);
                                    let result = match st.buffers.create_host(Arc::new(vec![])) {
                                        Some(handle) => {
                                            CompletionResult::Ok(SuccessPayload::Handle(handle))
                                        }
                                        None => CompletionResult::Err(CompError {
                                            code: COMP_ERR_GRANT_EXHAUSTED,
                                            detail: Some(
                                                "buffer quota exhausted (deny new buffers)".into(),
                                            ),
                                        }),
                                    };
                                    st.enqueue_completion(op, &result).map_err(|e| {
                                        Trap::bare(TrapCode::BadModule, e.to_string())
                                    })?;
                                    op
                                } else {
                                    let request = OpRequest::ArtifactRange {
                                        hash,
                                        range_off,
                                        range_len,
                                        span_off,
                                        span_len,
                                    };
                                    let op = st.ops.begin(request.clone()).map_err(|code| {
                                        Trap::new(
                                            code,
                                            "fetch",
                                            None,
                                            "max_outstanding grant (§2.3)",
                                        )
                                    })?;
                                    st.op_requests.push((op, request));
                                    op
                                }
                            }
                        }
                    }
                    // Plain artifact (manifest/tokenizer/module blob): the whole-artifact
                    // verify-then-slice path, with a definite-length request charged against
                    // the read budget at the call (a `range_len == 0` whole fetch charges at
                    // the artifact's true size only once known — plain artifacts are the small
                    // class; the shard volume is always chunk-registered and fully charged).
                    None => {
                        if budget != 0 && used.saturating_add(range_len) > budget {
                            immediate_fetch_refusal(
                                &mut st,
                                hash,
                                range_off,
                                range_len,
                                COMP_ERR_GRANT_EXHAUSTED,
                                format!(
                                    "data-read budget exhausted ({used} of {budget} bytes \
                                     used; {range_len} more requested)"
                                ),
                            )?
                        } else {
                            st.data_read_used += range_len;
                            let request = OpRequest::ArtifactFetch {
                                hash,
                                range_off,
                                range_len,
                            };
                            let op = st.ops.begin(request.clone()).map_err(|code| {
                                Trap::new(code, "fetch", None, "max_outstanding grant (§2.3)")
                            })?;
                            st.op_requests.push((op, request));
                            op
                        }
                    }
                };
                st.note_egress();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- data@2::register_chunks — the chunk-addressed corpus registration (minor 2) ------------
    // The module presents one shard's chunk map as canonical CBOR
    // `[chunk_size, token_count, byte_len, [c_0, …]]`; the host re-derives the domain-separated
    // fold and admits the map ONLY when the fold IS a granted artifact hash — a module cannot
    // register chunks for content it was not granted, and a lying chunk list can never derive a
    // granted identity. Deterministic guest output (§2.7 dc class): no journal record; replay
    // re-executes the registration over reproduced guest memory. Idempotent per identity.
    linker.func_wrap(
        NS_DATA_V2,
        "register_chunks",
        |mut c: Caller<'_, Host>, desc_ptr: u32, desc_len: u32| -> Result<u32, wasmtime::Error> {
            let r: Result<u32, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("register_chunks")?;
                let desc = read_guest(c, desc_ptr, desc_len)?;
                let map = decode_chunk_descriptor(&desc).map_err(|detail| {
                    Trap::new(TrapCode::BadEnum, "register_chunks", None, detail)
                })?;
                let fold = map.fold();
                if !c.data().granted_artifacts.contains(&fold.0) {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "register_chunks",
                        None,
                        format!(
                            "chunk-map fold {} is not in the admitted artifact set (which \
                             artifacts a module may touch is a grant, architecture §3.2)",
                            fold.to_hex()
                        ),
                    ));
                }
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.chunk_maps.insert(fold.0, map);
                Ok(0)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    Ok(())
}

/// Service a `data.fetch` of a **self-sealed state fold** ([SF-R1], ABI §12.14) host-locally at
/// the call: bounds + read-budget checks (state reads ride the existing `data-read-budget`,
/// [CC-5]), a uniform OpId mint (§7.1) retired at once, range assembly from the state store's
/// content-addressed chunks (per-chunk lengths — per-parameter chunking is not a uniform grid),
/// and an immediate journaled completion. The guest-visible protocol is byte-identical to an
/// embedder-serviced fetch; replay materializes the same completion from its replay-side state
/// chunk store (never the payload table — `ReplayMissingPayload` cannot fire for a self-sealed
/// root).
fn service_self_sealed_fetch(
    st: &mut PumpState,
    hash: [u8; 32],
    range_off: u64,
    range_len: u64,
) -> Result<u64, Trap> {
    let byte_len = st
        .state
        .sealed(&hash)
        .expect("caller checked sealed")
        .byte_len;
    let end = if range_len == 0 {
        byte_len
    } else {
        range_off.saturating_add(range_len)
    };
    if range_off > byte_len || end > byte_len {
        return immediate_fetch_refusal(
            st,
            hash,
            range_off,
            range_len,
            daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
            format!("range [{range_off}, {end}) out of bounds (sealed family is {byte_len} bytes)"),
        );
    }
    let charge = end - range_off;
    let (used, budget) = (st.data_read_used, st.data_read_budget);
    if budget != 0 && used.saturating_add(charge) > budget {
        return immediate_fetch_refusal(
            st,
            hash,
            range_off,
            range_len,
            COMP_ERR_GRANT_EXHAUSTED,
            format!(
                "data-read budget exhausted ({used} of {budget} bytes used; {charge} more \
                 requested)"
            ),
        );
    }
    st.data_read_used += charge;
    let op = st
        .ops
        .begin(OpRequest::ArtifactFetch {
            hash,
            range_off,
            range_len,
        })
        .map_err(|code| Trap::new(code, "fetch", None, "max_outstanding grant (§2.3)"))?;
    st.ops.finish(op);
    let result = match st
        .state
        .read_range(&hash, range_off, end)
        .expect("caller checked sealed")
    {
        Ok(bytes) => match st.buffers.create_host(Arc::new(bytes)) {
            Some(handle) => CompletionResult::Ok(SuccessPayload::Handle(handle)),
            None => CompletionResult::Err(CompError {
                code: COMP_ERR_GRANT_EXHAUSTED,
                detail: Some("buffer quota exhausted (deny new buffers)".into()),
            }),
        },
        Err(detail) => CompletionResult::Err(CompError {
            code: daemon_vhc_abi::COMP_ERR_HASH_MISMATCH,
            detail: Some(detail),
        }),
    };
    st.enqueue_completion(op, &result)
        .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
    Ok(op)
}

/// Mint + immediately refuse one `data.fetch` op with a typed completion error (bounds/budget
/// refusals whose facts are knowable at the call): the OpId mint stays uniform (§7.1 — every
/// OpId derives from the one `begin()` sequence), the op retires at once, and the journaled
/// tag-14 completion carries the refusal for replay.
fn immediate_fetch_refusal(
    st: &mut PumpState,
    hash: [u8; 32],
    range_off: u64,
    range_len: u64,
    code: u64,
    detail: String,
) -> Result<u64, Trap> {
    let op = st
        .ops
        .begin(OpRequest::ArtifactFetch {
            hash,
            range_off,
            range_len,
        })
        .map_err(|code| Trap::new(code, "fetch", None, "max_outstanding grant (§2.3)"))?;
    st.ops.finish(op);
    let result = CompletionResult::Err(CompError {
        code,
        detail: Some(detail),
    });
    st.enqueue_completion(op, &result)
        .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
    Ok(op)
}
