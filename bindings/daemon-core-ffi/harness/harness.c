/*
 * daemon-core-ffi C harness — the cross-language analogue of the in-process transcript test.
 *
 * Proves the C ABI is the *same* §17 session surface: create a runtime + session, submit a
 * CBOR-encoded AgentCommand::StartTurn, then poll the drain queue until a drained item carries the
 * "TurnFinished" event. It then drives the phase-9 control surface over the same ABI: a Snapshot
 * command (expecting a "Snapshot" event) and a Steer command (expecting a "Steered" event). The
 * CBOR bytes are pinned by the Rust `fixture_tests::*_fixture_matches_canonical_cbor` tests so they
 * cannot silently drift.
 *
 * Build + run via `harness/run.sh` (links the staticlib). Exit 0 on success, non-zero otherwise.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../include/daemon_core.h"

/* CBOR for AgentCommand::StartTurn { input: { text: "hi", attachments: [], notice: null },
 * request_id: 1 }. `notice` is the wire-v29 UserMsg field (serde default, always encoded). */
static const uint8_t START_TURN_HI[] = {
    0xA1,
    0x69, 'S','t','a','r','t','T','u','r','n',
    0xA2,
    0x65, 'i','n','p','u','t',
    0xA3,
    0x64, 't','e','x','t',
    0x62, 'h','i',
    0x6B, 'a','t','t','a','c','h','m','e','n','t','s',
    0x80,
    0x66, 'n','o','t','i','c','e',
    0xF6,
    0x6A, 'r','e','q','u','e','s','t','_','i','d',
    0x01,
};

/*
 * CBOR for a partial CoreFfiConfig { provider: "mock", system_prompt: "harness-cfg" } — the
 * construction-config blob `daemon_runtime_new_with_config` decodes. Every field is
 * `#[serde(default)]`, so this minimal map degrades to the zero-config mock brain with a custom
 * prompt (a real provider would set provider: "genai" + model + api_key). Pinned by the Rust
 * `config_tests::core_config_blob_decodes_to_mock` test so it cannot silently drift.
 */
static const uint8_t CORE_CONFIG[] = {
    0xA2,
    0x68, 'p','r','o','v','i','d','e','r',
    0x64, 'm','o','c','k',
    0x6D, 's','y','s','t','e','m','_','p','r','o','m','p','t',
    0x6B, 'h','a','r','n','e','s','s','-','c','f','g',
};

/* CBOR for AgentCommand::Snapshot { request_id: 2 }. */
static const uint8_t SNAPSHOT_2[] = {
    0xA1,
    0x68, 'S','n','a','p','s','h','o','t',
    0xA1,
    0x6A, 'r','e','q','u','e','s','t','_','i','d',
    0x02,
};

/* CBOR for AgentCommand::Steer { text: "go", request_id: 3 }. */
static const uint8_t STEER_GO[] = {
    0xA1,
    0x65, 'S','t','e','e','r',
    0xA2,
    0x64, 't','e','x','t',
    0x62, 'g','o',
    0x6A, 'r','e','q','u','e','s','t','_','i','d',
    0x03,
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

/*
 * Submit a CBOR command, then poll the drain queue until an item carries `needle`.
 * Returns 1 on success, 0 otherwise.
 */
static int submit_and_await(daemon_session_t *s, const uint8_t *cmd, size_t cmd_len,
                            const char *what, const char *needle) {
    int rc = daemon_session_submit(s, cmd, cmd_len);
    if (rc != DAEMON_OK) { print_last_error(what); return 0; }

    uint8_t buf[4096];
    struct timespec nap = { .tv_sec = 0, .tv_nsec = 10 * 1000 * 1000 }; /* 10ms */
    for (int i = 0; i < 500; i++) {
        size_t out_len = 0;
        int pr = daemon_session_poll(s, buf, sizeof(buf), &out_len);
        if (pr == DAEMON_OK) {
            if (contains(buf, out_len, needle)) return 1;
        } else if (pr == DAEMON_EMPTY) {
            nanosleep(&nap, NULL);
        } else if (pr == DAEMON_BUFFER_TOO_SMALL) {
            fprintf(stderr, "%s poll: item needs %zu bytes\n", what, out_len);
            return 0;
        } else {
            print_last_error(what);
            return 0;
        }
    }
    fprintf(stderr, "FAIL: %s never observed %s\n", what, needle);
    return 0;
}

int main(void) {
    printf("daemon-core-ffi harness: abi_version=%u\n", daemon_abi_version());

    daemon_runtime_t *rt = daemon_runtime_new();
    if (!rt) { print_last_error("runtime_new"); return 1; }

    const char *name = "ffi-session";
    daemon_session_t *s = daemon_session_open(rt, (const uint8_t *)name, strlen(name));
    if (!s) { print_last_error("session_open"); daemon_runtime_free(rt); return 1; }

    int ok = 1;
    if (ok && submit_and_await(s, START_TURN_HI, sizeof(START_TURN_HI), "start_turn", "TurnFinished")) {
        printf("OK: drained TurnFinished over the C ABI\n");
    } else { ok = 0; }

    /* The phase-9 control surface, over the same ABI. */
    if (ok && submit_and_await(s, SNAPSHOT_2, sizeof(SNAPSHOT_2), "snapshot", "Snapshot")) {
        printf("OK: drained Snapshot over the C ABI\n");
    } else { ok = 0; }

    if (ok && submit_and_await(s, STEER_GO, sizeof(STEER_GO), "steer", "Steered")) {
        printf("OK: drained Steered over the C ABI\n");
    } else { ok = 0; }

    daemon_session_free(s);
    daemon_runtime_free(rt);

    /*
     * The construction-config entry point, over the same ABI: stand a runtime up from a CBOR
     * CoreFfiConfig and drive the same transcript. (This blob keeps the mock provider so the
     * harness stays network-free; a real embedder sets provider: "genai" + model + api_key.)
     */
    if (ok) {
        daemon_runtime_t *crt = daemon_runtime_new_with_config(CORE_CONFIG, sizeof(CORE_CONFIG));
        if (!crt) { print_last_error("runtime_new_with_config"); return 1; }
        const char *cname = "ffi-cfg-session";
        daemon_session_t *cs = daemon_session_open(crt, (const uint8_t *)cname, strlen(cname));
        if (!cs) { print_last_error("cfg session_open"); daemon_runtime_free(crt); return 1; }
        if (submit_and_await(cs, START_TURN_HI, sizeof(START_TURN_HI), "cfg start_turn", "TurnFinished")) {
            printf("OK: drained TurnFinished from a config-built runtime\n");
        } else { ok = 0; }
        daemon_session_free(cs);
        daemon_runtime_free(crt);
    }

    if (ok) {
        printf("OK: C ABI exercised StartTurn + Snapshot + Steer + config runtime\n");
        return 0;
    }
    return 1;
}
