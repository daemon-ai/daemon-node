#ifndef DAEMON_H
#define DAEMON_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

/**
 * `daemon_status`: the success/failure code every entry point returns.
 */
#define DAEMON_OK 0

/**
 * A recoverable error occurred; details via `daemon_last_error`.
 */
#define DAEMON_ERROR 1

/**
 * A panic was caught at the boundary (should not happen in normal operation).
 */
#define DAEMON_PANIC 3

/**
 * A null handle or invalid argument was passed.
 */
#define DAEMON_INVALID 4

/**
 * Opaque durable-host handle: owns the Tokio runtime, the assembled node surface, and its started
 * resident-service supervisor (taken on `daemon_host_free` to drive a graceful shutdown).
 */
typedef struct daemon_host_t daemon_host_t;

/**
 * The ABI version of this shell.
 */
uint32_t daemon_abi_version(void);

/**
 * Boot a durable host with the zero-config default (in-memory store, deterministic mock provider).
 * Returns null on failure (see `daemon_last_error`). Use `daemon_host_new_with_config` for a real
 * store/provider.
 */
struct daemon_host_t *daemon_host_new(void);

/**
 * Boot a durable host from a CBOR-encoded [`HostFfiConfig`] `(cfg, len)`. Returns null on failure
 * (see `daemon_last_error`).
 *
 * # Safety
 * `cfg` must point to `len` readable bytes (a CBOR `HostFfiConfig`); `len` may be `0` for defaults.
 */
struct daemon_host_t *daemon_host_new_with_config(const uint8_t *cfg, size_t len);

/**
 * Free a host handle created by `daemon_host_new`/`daemon_host_new_with_config`, gracefully
 * shutting its resident services down first.
 *
 * # Safety
 * `h` must be a pointer returned by a `daemon_host_new*` and not already freed.
 */
void daemon_host_free(struct daemon_host_t *h);

/**
 * Dispatch a CBOR-encoded [`ApiRequest`] against the node and return the CBOR-encoded
 * [`ApiResponse`]. On `DAEMON_OK`, `*out_resp` points to a library-owned buffer of `*out_len`
 * bytes that the caller releases with `daemon_buf_free` (callee-allocates / callee-frees,
 * daemon-ffi-spec §3.1).
 *
 * This one call carries the entire node surface — `Submit`/`Poll`/`Respond`, fleet/tree, fs/cron,
 * model/profile/credential/auth — exactly as the Unix-socket transport routes it.
 *
 * # Safety
 * `h` must be valid; `req` must point to `req_len` readable bytes; `out_resp` and `out_len` must be
 * writable.
 */
int32_t daemon_host_call(struct daemon_host_t *h,
                         const uint8_t *req,
                         size_t req_len,
                         uint8_t **out_resp,
                         size_t *out_len);

/**
 * Copy the thread-local last-error message (UTF-8, not NUL-terminated) into `buf`, writing its full
 * length into `out_len`. Returns `DAEMON_OK`.
 *
 * # Safety
 * `buf` must point to `cap` writable bytes; `out_len` must be writable.
 */
int32_t daemon_last_error(uint8_t *buf, size_t cap, size_t *out_len);

/**
 * Free a library-allocated buffer `(ptr, len)` handed out by `daemon_host_call`.
 *
 * # Safety
 * `(ptr, len)` must be a buffer previously returned by `daemon_host_call`, not already freed.
 */
void daemon_buf_free(uint8_t *ptr, size_t len);

#endif  /* DAEMON_H */
