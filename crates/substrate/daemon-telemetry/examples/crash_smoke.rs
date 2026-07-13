// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Throwaway crash-reporting smoke: arms the reporter (env-gated) then dies with SIGSEGV.
//! Run against a LOCAL mock ingest endpoint only — see the validation notes in the plan.

fn main() {
    let _crash = daemon_telemetry::init_crash_reporting("crash-smoke");
    assert!(
        _crash.is_armed(),
        "smoke requires DAEMON_SENTRY_DSN + DAEMON_CRASH_CONSENT=1"
    );
    eprintln!("crash-smoke: armed; triggering SIGSEGV in 1s");
    std::thread::sleep(std::time::Duration::from_secs(1));
    unsafe {
        let p: *mut u32 = std::ptr::null_mut();
        std::ptr::write_volatile(p, 42);
    }
    unreachable!("the write above must fault");
}
