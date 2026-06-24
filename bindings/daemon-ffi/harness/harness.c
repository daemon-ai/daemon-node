/*
 * daemon-ffi C harness — the cross-language analogue of the durable-node transcript test.
 *
 * Proves the C ABI drives the *durable host* over one generic call: boot an in-memory node
 * (daemon_host_new), then marshal CBOR ApiRequests through daemon_host_call — an
 * ApiRequest::Submit { StartTurn } followed by ApiRequest::Poll until a drained ApiResponse carries
 * the "TurnFinished" event. The CBOR request bytes are pinned by the Rust
 * `wire_fixtures::*_matches_canonical_cbor` tests so they cannot silently drift.
 *
 * Build + run via `harness/run.sh` (links the staticlib). Exit 0 on success, non-zero otherwise.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../include/daemon.h"

/*
 * CBOR for ApiRequest::Submit {
 *   session: "ffi-host", command: StartTurn { input: { text: "hi", attachments: [] },
 *   request_id: 1 }, origin: null, profile: null
 * }.
 */
static const uint8_t SUBMIT_START_TURN[] = {
    0xA1,
    0x66, 'S','u','b','m','i','t',
    0xA4,
    0x67, 's','e','s','s','i','o','n',
    0x68, 'f','f','i','-','h','o','s','t',
    0x67, 'c','o','m','m','a','n','d',
    0xA1,
    0x69, 'S','t','a','r','t','T','u','r','n',
    0xA2,
    0x65, 'i','n','p','u','t',
    0xA2,
    0x64, 't','e','x','t',
    0x62, 'h','i',
    0x6B, 'a','t','t','a','c','h','m','e','n','t','s',
    0x80,
    0x6A, 'r','e','q','u','e','s','t','_','i','d',
    0x01,
    0x66, 'o','r','i','g','i','n',
    0xF6, /* null */
    0x67, 'p','r','o','f','i','l','e',
    0xF6, /* null */
};

/* CBOR for ApiRequest::Poll { session: "ffi-host", max: 16 }. */
static const uint8_t POLL[] = {
    0xA1,
    0x64, 'P','o','l','l',
    0xA2,
    0x67, 's','e','s','s','i','o','n',
    0x68, 'f','f','i','-','h','o','s','t',
    0x63, 'm','a','x',
    0x10, /* 16 */
};

/* Naive byte-substring search (the CBOR carries the variant name as a text string). */
static int contains(const uint8_t *hay, size_t hay_len, const char *needle) {
    size_t n = strlen(needle);
    if (n > hay_len) return 0;
    for (size_t i = 0; i + n <= hay_len; i++) {
        if (memcmp(hay + i, needle, n) == 0) return 1;
    }
    return 0;
}

static void print_last_error(const char *ctx) {
    uint8_t buf[512];
    size_t len = 0;
    daemon_last_error(buf, sizeof(buf), &len);
    if (len > sizeof(buf)) len = sizeof(buf);
    fprintf(stderr, "%s: %.*s\n", ctx, (int)len, (char *)buf);
}

/* Call the node with a CBOR ApiRequest; returns 1 and (optionally) tests the response for `needle`. */
static int call(daemon_host_t *h, const uint8_t *req, size_t req_len,
                const char *what, const char *needle, int *found) {
    uint8_t *resp = NULL;
    size_t resp_len = 0;
    int rc = daemon_host_call(h, req, req_len, &resp, &resp_len);
    if (rc != DAEMON_OK) { print_last_error(what); return 0; }
    if (needle && found) {
        *found = contains(resp, resp_len, needle);
    }
    daemon_buf_free(resp, resp_len);
    return 1;
}

int main(void) {
    printf("daemon-ffi harness: abi_version=%u\n", daemon_abi_version());

    daemon_host_t *h = daemon_host_new();
    if (!h) { print_last_error("host_new"); return 1; }

    int ok = 1;
    if (ok && !call(h, SUBMIT_START_TURN, sizeof(SUBMIT_START_TURN), "submit", NULL, NULL)) {
        ok = 0;
    } else if (ok) {
        printf("OK: submitted StartTurn over the durable node call\n");
    }

    /* Poll the drain until the mock node's terminal event arrives. */
    int finished = 0;
    struct timespec nap = { .tv_sec = 0, .tv_nsec = 20 * 1000 * 1000 }; /* 20ms */
    for (int i = 0; ok && i < 200 && !finished; i++) {
        if (!call(h, POLL, sizeof(POLL), "poll", "TurnFinished", &finished)) { ok = 0; break; }
        if (!finished) nanosleep(&nap, NULL);
    }
    if (ok && finished) {
        printf("OK: drained TurnFinished over the durable node call\n");
    } else if (ok) {
        fprintf(stderr, "FAIL: never drained TurnFinished\n");
        ok = 0;
    }

    daemon_host_free(h);

    if (ok) {
        printf("OK: C ABI booted a durable node and drove a turn\n");
        return 0;
    }
    return 1;
}
