#ifndef DAEMON_CORE_H
#define DAEMON_CORE_H

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
 * No item was available to drain (poll on an idle session).
 */
#define DAEMON_EMPTY 2

/**
 * A panic was caught at the boundary (should not happen in normal operation).
 */
#define DAEMON_PANIC 3

/**
 * A null handle or invalid argument was passed.
 */
#define DAEMON_INVALID 4

/**
 * The caller buffer was too small for the next item.
 */
#define DAEMON_BUFFER_TOO_SMALL 5

/**
 * Opaque runtime handle: owns the Tokio runtime and the session-surface implementation.
 */
typedef struct daemon_runtime_t daemon_runtime_t;

/**
 * Opaque session handle: a runtime handle + the session-surface impl + a bound session id.
 */
typedef struct daemon_session_t daemon_session_t;

/**
 * The ABI version of this shell.
 */
uint32_t daemon_abi_version(void);

/**
 * Create a runtime handle with the zero-config mock brain (deterministic provider, embedded L1
 * credential pool, no journal). Returns null on failure (see `daemon_last_error`). Use
 * `daemon_runtime_new_with_config` to drive a real provider.
 */
struct daemon_runtime_t *daemon_runtime_new(void);

/**
 * Create a runtime handle from a CBOR-encoded [`CoreFfiConfig`] `(cfg, len)` — the seam that wires
 * a *real* provider and an injected API key into every session's engine. Returns null on failure
 * (see `daemon_last_error`).
 *
 * # Safety
 * `cfg` must point to `len` readable bytes (a CBOR `CoreFfiConfig`); `len` may be `0` for the
 * default config.
 */
struct daemon_runtime_t *daemon_runtime_new_with_config(const uint8_t *cfg,
                                                        size_t len);

/**
 * Free a runtime handle created by `daemon_runtime_new`.
 *
 * # Safety
 * `rt` must be a pointer returned by `daemon_runtime_new` and not already freed.
 */
void daemon_runtime_free(struct daemon_runtime_t *rt);

/**
 * Open a session bound to `rt`, identified by the UTF-8 name `(name, name_len)`. Returns null on
 * failure.
 *
 * # Safety
 * `rt` must be valid; `name` must point to `name_len` readable bytes.
 */
struct daemon_session_t *daemon_session_open(struct daemon_runtime_t *rt,
                                             const uint8_t *name,
                                             size_t name_len);

/**
 * Free a session handle created by `daemon_session_open`.
 *
 * # Safety
 * `s` must be a pointer returned by `daemon_session_open` and not already freed.
 */
void daemon_session_free(struct daemon_session_t *s);

/**
 * Submit a CBOR-encoded `AgentCommand` to the session.
 *
 * # Safety
 * `s` must be valid; `cmd` must point to `len` readable bytes.
 */
int32_t daemon_session_submit(struct daemon_session_t *s, const uint8_t *cmd, size_t len);

/**
 * Drain the next outbound item (CBOR-encoded [`daemon_api::Outbound`]) into the caller buffer.
 * Returns `DAEMON_EMPTY` when idle, `DAEMON_BUFFER_TOO_SMALL` if `cap` is too small (and writes the
 * needed length into `out_len`).
 *
 * # Safety
 * `s` must be valid; `out_buf` must point to `cap` writable bytes; `out_len` must be writable.
 */
int32_t daemon_session_poll(struct daemon_session_t *s,
                            uint8_t *out_buf,
                            size_t cap,
                            size_t *out_len);

/**
 * Answer a parked host request with a CBOR-encoded `HostResponse` (its `request_id` correlates).
 *
 * # Safety
 * `s` must be valid; `resp` must point to `len` readable bytes.
 */
int32_t daemon_session_respond(struct daemon_session_t *s, const uint8_t *resp, size_t len);

/**
 * Copy the thread-local last-error message (UTF-8, not NUL-terminated) into `buf`, writing its full
 * length into `out_len`. Returns `DAEMON_OK`.
 *
 * # Safety
 * `buf` must point to `cap` writable bytes; `out_len` must be writable.
 */
int32_t daemon_last_error(uint8_t *buf, size_t cap, size_t *out_len);

/**
 * Free a library-allocated byte buffer `(ptr, len)`. Provided for the callee-allocates ownership
 * convention (daemon-ffi-spec §3.1); the poll path uses caller buffers and does not require it.
 *
 * # Safety
 * `(ptr, len)` must be a buffer previously handed out by this library, not already freed.
 */
void daemon_buf_free(uint8_t *ptr, size_t len);

#endif  /* DAEMON_CORE_H */
