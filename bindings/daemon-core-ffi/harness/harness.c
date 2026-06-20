/*
 * daemon-core-ffi C harness — the cross-language analogue of the in-process transcript test.
 *
 * Proves the C ABI is the *same* §17 session surface: create a runtime + session, submit a
 * CBOR-encoded AgentCommand::StartTurn, then poll the drain queue until a drained item carries the
 * "TurnFinished" event. The CBOR bytes are pinned by the Rust test
 * `fixture_tests::start_turn_fixture_matches_canonical_cbor` so they cannot silently drift.
 *
 * Build + run via `harness/run.sh` (links the staticlib). Exit 0 on success, non-zero otherwise.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../include/daemon_core.h"

/* CBOR for AgentCommand::StartTurn { input: { text: "hi" }, request_id: 1 }. */
static const uint8_t START_TURN_HI[] = {
    0xA1,
    0x69, 'S','t','a','r','t','T','u','r','n',
    0xA2,
    0x65, 'i','n','p','u','t',
    0xA1,
    0x64, 't','e','x','t',
    0x62, 'h','i',
    0x6A, 'r','e','q','u','e','s','t','_','i','d',
    0x01,
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

int main(void) {
    printf("daemon-core-ffi harness: abi_version=%u\n", daemon_abi_version());

    daemon_runtime_t *rt = daemon_runtime_new();
    if (!rt) { print_last_error("runtime_new"); return 1; }

    const char *name = "ffi-session";
    daemon_session_t *s = daemon_session_open(rt, (const uint8_t *)name, strlen(name));
    if (!s) { print_last_error("session_open"); daemon_runtime_free(rt); return 1; }

    int rc = daemon_session_submit(s, START_TURN_HI, sizeof(START_TURN_HI));
    if (rc != DAEMON_OK) { print_last_error("submit"); daemon_session_free(s); daemon_runtime_free(rt); return 1; }

    int finished = 0;
    uint8_t buf[4096];
    struct timespec nap = { .tv_sec = 0, .tv_nsec = 10 * 1000 * 1000 }; /* 10ms */
    for (int i = 0; i < 500 && !finished; i++) {
        size_t out_len = 0;
        int pr = daemon_session_poll(s, buf, sizeof(buf), &out_len);
        if (pr == DAEMON_OK) {
            if (contains(buf, out_len, "TurnFinished")) {
                finished = 1;
                break;
            }
        } else if (pr == DAEMON_EMPTY) {
            nanosleep(&nap, NULL);
        } else if (pr == DAEMON_BUFFER_TOO_SMALL) {
            fprintf(stderr, "poll: item needs %zu bytes\n", out_len);
            break;
        } else {
            print_last_error("poll");
            break;
        }
    }

    daemon_session_free(s);
    daemon_runtime_free(rt);

    if (finished) {
        printf("OK: drained TurnFinished over the C ABI\n");
        return 0;
    }
    fprintf(stderr, "FAIL: never observed TurnFinished\n");
    return 1;
}
