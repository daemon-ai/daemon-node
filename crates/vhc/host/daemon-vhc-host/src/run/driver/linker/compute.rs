// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `compute@2` world (track C1, ABI §15; architecture §3.3/§3.4) — the Burn-IR command
//! queue: the wire is CBOR(`burn_ir::OperationIr`) at the pinned Burn version; dispatch is the
//! `ComputeRunner` (burn-router runner + typed handle faults + the deferred-error latch).
//! Validation faults trap at the call (§7.6 programming errors); DEVICE faults defer to fence
//! (`ComputeFault` trap) / export (`COMP_ERR_DEVICE` completion) — §3.3.

use std::sync::Arc;

use wasmtime::{Caller, Linker};

use daemon_vhc_abi::{COMP_ERR_GRANT_EXHAUSTED, NS_COMPUTE_V2};

use crate::run::completion::{CompError, CompletionResult, SuccessPayload};
use crate::run::driver::host::{read_guest, stash, Host};
use crate::run::ops::OpRequest;
use crate::trap::{Trap, TrapCode};

/// Link the `compute@2` imports.
#[allow(clippy::too_many_lines)]
pub(super) fn link(linker: &mut Linker<Host>) -> Result<(), wasmtime::Error> {
    // ==== compute@2 (track C1, ABI §15; architecture §3.3/§3.4) — the Burn-IR command queue ======
    // The wire is CBOR(burn_ir::OperationIr) at the pinned Burn version; dispatch is the
    // ComputeRunner (burn-router runner + typed handle faults + the deferred-error latch).
    // Validation faults trap at the call (§7.6 programming errors); DEVICE faults defer to
    // fence (ComputeFault trap) / export (COMP_ERR_DEVICE completion) — §3.3.

    // ---- compute@2::submit_op — enqueue one op-blob (infallible for device faults) --------------
    linker.func_wrap(
        NS_COMPUTE_V2,
        "submit_op",
        |mut c: Caller<'_, Host>, op_ptr: u32, op_len: u32| -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("submit_op")?;
                let op_cbor = read_guest(c, op_ptr, op_len)?;
                let d = c.data_mut();
                if d.compute.is_none() {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                }
                // The queue-depth grant (architecture §3.3): outstanding device work is bounded;
                // the guest reclaims depth by fencing.
                if d.compute_queue_depth != 0 && d.compute_ops_since_fence >= d.compute_queue_depth
                {
                    return Err(Trap::new(
                        TrapCode::GrantViolation,
                        "submit_op",
                        None,
                        format!(
                            "compute queue depth {} reached — fence to reclaim (§3.3)",
                            d.compute_queue_depth
                        ),
                    ));
                }
                let compute = d.compute.as_mut().expect("checked above");
                compute
                    .submit_op(&op_cbor)
                    .map_err(|e| Trap::new(e.trap_code(), "submit_op", None, e.to_string()))?;
                d.compute_ops_since_fence += 1;
                d.compute_ops_total += 1;
                // The deferred-fault injection seam (RunConfig::compute_fault_after_ops).
                if d.compute_fault_after_ops == Some(d.compute_ops_total - 1) {
                    d.compute
                        .as_mut()
                        .expect("checked above")
                        .inject_device_fault("injected deferred device fault (test seam)");
                }
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- compute@2::fence — insert a marker; Event::Fence(id) delivers when the device passes it
    linker.func_wrap(
        NS_COMPUTE_V2,
        "fence",
        |mut c: Caller<'_, Host>, fence_id: u64| -> Result<(), wasmtime::Error> {
            let r: Result<(), Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("fence")?;
                let d = c.data_mut();
                let Some(compute) = d.compute.as_mut() else {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                };
                // Deferred device errors surface HERE, typed (§3.3): the fence event is
                // delivered only on a successful drain, so a delivered Fence is a real
                // consistency point.
                compute
                    .fence()
                    .map_err(|e| Trap::new(e.trap_code(), "fence", None, e.to_string()))?;
                d.compute_ops_since_fence = 0;
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                st.enqueue_fence(fence_id)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                drop(st);
                shared.wake.notify_all();
                Ok(())
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- compute@2::export — device tensor → sealed buffer (bulk bytes ride the BufferHandle,
    // §3.4 — never inline in the op-stream). Returns an OpId; completes Ok(BufferHandle) with the
    // CBOR(TensorData), journaled verbatim (kind-5 tag-2 — device bytes are a nondeterministic
    // input); a deferred device error completes Err(COMP_ERR_DEVICE) — the readback twin of the
    // fence trap.
    linker.func_wrap(
        NS_COMPUTE_V2,
        "export",
        |mut c: Caller<'_, Host>, ir_ptr: u32, ir_len: u32| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("export")?;
                let ir_cbor = read_guest(c, ir_ptr, ir_len)?;
                let d = c.data_mut();
                let Some(compute) = d.compute.as_mut() else {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                };
                // Stale/invalid handles and undecodable IR are programming errors → trap at the
                // call (§7.6); only the DEVICE fault defers into the completion.
                let read = match compute.read_tensor(&ir_cbor) {
                    Ok(data) => Ok(data),
                    Err(e @ crate::compute::ComputeError::Device(_)) => Err(e),
                    Err(e) => return Err(Trap::new(e.trap_code(), "export", None, e.to_string())),
                };
                let shared = c.data().shared.clone();
                let mut st = shared.state.lock().expect("pump lock");
                let op = st.ops.begin(OpRequest::TensorExport).map_err(|code| {
                    Trap::new(code, "export", None, "max_outstanding grant (§2.3)")
                })?;
                // Pump-internal service AT THE CALL (the runner is host-local): the op never
                // reaches `op_requests`, so transport seats never see a TensorExport.
                st.ops.finish(op);
                let result = match read {
                    Ok(data) => {
                        // Journal the device bytes verbatim (kind 5) BEFORE the completion
                        // record — the stream-read (kind 4) discipline: replay materializes the
                        // completion's buffer from this record and re-executes no kernel (§8.7).
                        st.sink
                            .read_back(
                                op,
                                u64::from(daemon_vhc_abi::READBACK_KIND_TENSOR_EXPORT),
                                daemon_vhc_abi::RET_STATUS_DELIVERED,
                                &data,
                            )
                            .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                        match st.buffers.create_host(Arc::new(data)) {
                            Some(handle) => CompletionResult::Ok(SuccessPayload::Handle(handle)),
                            None => CompletionResult::Err(CompError {
                                code: COMP_ERR_GRANT_EXHAUSTED,
                                detail: Some("buffer quota exhausted (deny new buffers)".into()),
                            }),
                        }
                    }
                    Err(e) => CompletionResult::Err(CompError {
                        code: daemon_vhc_abi::COMP_ERR_DEVICE,
                        detail: Some(e.to_string()),
                    }),
                };
                st.enqueue_completion(op, &result)
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                drop(st);
                shared.wake.notify_all();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    // ---- compute@2::import — sealed buffer → device tensor under the guest-minted TensorId.
    // Returns an OpId; completes Ok(()). Deterministic (guest bytes by way of the sealed buffer):
    // no journal record beyond the tag-14 completion.
    linker.func_wrap(
        NS_COMPUTE_V2,
        "import",
        |mut c: Caller<'_, Host>, buffer: u64, tensor_id: u64| -> Result<u64, wasmtime::Error> {
            let r: Result<u64, Trap> = (|c: &mut Caller<'_, Host>| {
                c.data_mut().enter("import")?;
                let shared = c.data().shared.clone();
                let bytes = {
                    let st = shared.state.lock().expect("pump lock");
                    st.buffers
                        .resolve(buffer)
                        .map_err(|code| Trap::new(code, "import", None, "buffer handle"))?
                };
                let d = c.data_mut();
                let Some(compute) = d.compute.as_mut() else {
                    return Err(Trap::bare(
                        TrapCode::BadModule,
                        "compute@2 import without a compute runner (linker invariant)",
                    ));
                };
                // The buffer must hold decodable CBOR(TensorData) — a malformed import is a
                // programming error at the call (§7.6), not a completion error.
                compute
                    .import_tensor(tensor_id, &bytes)
                    .map_err(|e| Trap::new(e.trap_code(), "import", None, e.to_string()))?;
                let mut st = shared.state.lock().expect("pump lock");
                let op = st
                    .ops
                    .begin(OpRequest::TensorImport { tensor_id })
                    .map_err(|code| {
                        Trap::new(code, "import", None, "max_outstanding grant (§2.3)")
                    })?;
                st.ops.finish(op); // pump-internal service at the call (see export)
                st.enqueue_completion(op, &CompletionResult::Ok(SuccessPayload::Unit))
                    .map_err(|e| Trap::bare(TrapCode::BadModule, e.to_string()))?;
                drop(st);
                shared.wake.notify_all();
                Ok(op)
            })(&mut c);
            stash(&mut c, r)
        },
    )?;

    Ok(())
}
