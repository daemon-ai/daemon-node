// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//! The device-lane panic recorder ([`daemon_vhc_host::device_panic`]): the seam that lets a device
//! failure reach the run as a TYPED fault carrying what the device actually said.
//!
//! Its own binary rather than a `#[cfg(test)]` module, and deliberately so: the recorder is
//! PROCESS-global (a panic on a device runner thread has to be readable from the guest thread, and
//! the two share nothing else), so a sibling test panicking in another thread lands in whatever
//! capture window happens to be open. In the crate's 80-odd-test unit binary that is a real race;
//! here the recorder has the process to itself and the assertions are exact.

use std::sync::{Mutex, MutexGuard, PoisonError};

use daemon_vhc_host::device_panic::{catch, take_report, Window};

/// The recorder is process-global (see the module docs), so these cases take turns: two of them
/// panicking at once would each land in the other's window, which is a property of the harness, not
/// of the recorder.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

#[test]
fn a_caught_panic_carries_its_message_and_location() {
    let _guard = serial();
    let err = catch(true, || panic!("the adapter is not there")).unwrap_err();
    assert!(err.contains("the adapter is not there"), "got {err}");
    assert!(
        err.contains("device_panic_capture.rs"),
        "the source location rides along, which the raw `Box<dyn Any>` payload cannot carry: {err}"
    );
}

/// The load-bearing one: a panic that ANOTHER stack caught and discarded — on another thread,
/// exactly like a cubecl device task — is still recoverable, so the primary device error can be
/// reported instead of whatever secondary symptom (a poisoned lock) surfaces next.
#[test]
fn a_panic_swallowed_on_another_thread_is_still_recoverable() {
    let _guard = serial();
    let _ = take_report();
    // A capture window is open (a device call is in flight on this thread) while the shape cubecl
    // produces happens elsewhere: the task's panic is caught by the library and its payload — a
    // `Box<dyn Any>`, which Debug-prints as the useless `Any { .. }` — never leaves that frame.
    let window = Window::open(true);
    std::thread::spawn(|| {
        let _ = std::panic::catch_unwind(|| panic!("device task exploded"));
    })
    .join()
    .expect("the worker thread itself does not unwind");
    drop(window);

    let report = take_report().expect("the discarded panic was recorded");
    assert!(report.contains("device task exploded"), "got {report}");
}

#[test]
fn a_clean_call_records_nothing() {
    let _guard = serial();
    let _ = take_report();
    assert_eq!(catch(false, || 7).unwrap(), 7);
    assert!(
        take_report().is_none(),
        "a call that did not panic must leave the slot empty, or every later device call would \
         report a stale fault"
    );
}

/// Nothing is recorded outside a capture window: the recorder is armed for the whole process, so
/// an unrelated panic elsewhere must not become the next device call's "cause".
#[test]
fn a_panic_outside_a_window_is_not_recorded() {
    let _guard = serial();
    let _ = take_report();
    std::thread::spawn(|| {
        let _ = std::panic::catch_unwind(|| panic!("unrelated"));
    })
    .join()
    .expect("the worker thread itself does not unwind");
    assert!(take_report().is_none());
}
