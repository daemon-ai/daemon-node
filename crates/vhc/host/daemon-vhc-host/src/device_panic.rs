// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Panic capture for the device lanes — so a device failure reaches the run as a TYPED fault
//! carrying what the device actually said (ABI §7.6/§15), instead of as a secondary symptom.
//!
//! # The problem this exists for
//!
//! The device stacks under this crate report failures by panicking, and then swallow the panic:
//!
//! * **cubecl runs each device task under its own `catch_unwind`** and logs the caught payload as
//!   `Task failed: {err:?}` — where `err` is a `Box<dyn Any>`, whose `Debug` is the literal text
//!   `Any { .. }`. The message and location are discarded at that point; nothing downstream can
//!   recover them.
//! * **The backend router's context sits behind a `Mutex`.** A task that panics while holding it
//!   poisons it, so the NEXT router call `unwrap()`s a `PoisonError` and panics with
//!   "poisoned lock: another task failed inside" — a message about the lock, not about the device.
//! * **This crate's own probe idiom silenced the hook** (`set_hook(Box::new(|_| {}))`) around
//!   expected bring-up failures, which suppressed the primary panic's report for every thread for
//!   the duration — including the device runner thread's.
//!
//! Composed, those three turn "this adapter does not exist" into a poisoned-lock panic on the guest
//! thread with no mention of the device, which is exactly how a fleet trainer died with its real
//! cause unrecoverable from the product's own output.
//!
//! # What this module does
//!
//! It installs ONE process-wide panic hook that **records** every panic's message + location into a
//! global slot and then delegates to the previous hook for the usual report. Two consequences:
//!
//! * A panic another library caught and discarded — on any thread, including a device runner
//!   thread — is still recoverable here ([`take_report`]), so the primary device error can be
//!   surfaced as the typed fault instead of the secondary symptom.
//! * The "expected failure" probes no longer need to blind the whole process: [`catch`] takes a
//!   `quiet` flag that suppresses only the REPORT (the recording always happens), and only for the
//!   duration of that call.
//!
//! The hook chains rather than replaces, so a host that installed its own reporter (the worker's
//! crash reporter) keeps it.

use std::panic::{AssertUnwindSafe, UnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// The most recent panic report (message + location) recorded inside an open capture window, from
/// ANY thread.
static LAST_REPORT: Mutex<Option<String>> = Mutex::new(None);

/// Nesting depth of open capture windows; the hook records only while this is non-zero, so a panic
/// with nothing to do with a device call never lands in the slot.
static CAPTURE_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Nesting depth of `quiet` capture windows; while non-zero the recorder does not delegate to the
/// previous hook (the panic is expected and handled by the caller).
static QUIET_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Install the recording hook exactly once per process.
fn arm() {
    static ARMED: OnceLock<()> = OnceLock::new();
    ARMED.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if CAPTURE_DEPTH.load(Ordering::Relaxed) > 0 {
                if let Ok(mut slot) = LAST_REPORT.lock() {
                    *slot = Some(info.to_string());
                }
            }
            if QUIET_DEPTH.load(Ordering::Relaxed) == 0 {
                previous(info);
            }
        }));
    });
}

/// An open capture window: while one is held, panics on ANY thread are recorded, and (when `quiet`)
/// their default report is held back.
pub struct Window {
    quiet: bool,
}

impl Window {
    #[must_use]
    pub fn open(quiet: bool) -> Self {
        arm();
        CAPTURE_DEPTH.fetch_add(1, Ordering::Relaxed);
        if quiet {
            QUIET_DEPTH.fetch_add(1, Ordering::Relaxed);
        }
        Self { quiet }
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        if self.quiet {
            QUIET_DEPTH.fetch_sub(1, Ordering::Relaxed);
        }
        CAPTURE_DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Take (and clear) the most recent recorded panic report.
///
/// This is the seam that recovers a panic some library caught and threw away: after a device call
/// that returned "successfully" but logged a swallowed task failure, a `Some` here IS that failure,
/// with the message and source location the `Box<dyn Any>` had already lost.
pub fn take_report() -> Option<String> {
    arm();
    LAST_REPORT.lock().ok().and_then(|mut slot| slot.take())
}

/// Run `f` under `catch_unwind`, returning its value or the panic's text.
///
/// `quiet` suppresses the default report for a panic the caller EXPECTS and handles (an adapter
/// probe that finds no adapter); the recording happens either way, so nothing is lost. A panic
/// raised while a quiet window is open on another thread is still recorded — it is only its console
/// report that is held back, and only for that window.
///
/// The returned text prefers the recorded report (message + location) over the raw payload, because
/// a `Box<dyn Any>` downcast yields the message alone and often not even that.
pub fn catch<T>(quiet: bool, f: impl FnOnce() -> T + UnwindSafe) -> Result<T, String> {
    arm();
    let _ = take_report();
    let window = Window::open(quiet);
    let outcome = std::panic::catch_unwind(f);
    drop(window);
    outcome.map_err(|payload| {
        take_report().unwrap_or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked with an unrecoverable payload".to_string())
        })
    })
}

/// [`catch`] for a `&mut`-capturing closure (the device-runner call sites): the runner is not
/// `UnwindSafe`, and it does not need to be — a caught device panic ends the run typed, so no
/// caller observes the runner afterwards.
pub fn catch_mut<T>(quiet: bool, f: impl FnOnce() -> T) -> Result<T, String> {
    catch(quiet, AssertUnwindSafe(f))
}

/// Run a device-runner call and fold BOTH failure shapes into one typed device fault:
///
/// * the call unwound → the panic's recorded text;
/// * the call returned, but a panic was recorded meanwhile → a device task failed and its stack
///   swallowed the panic (cubecl's `Task failed: Any { .. }`), so the "successful" return is not
///   evidence of health. This is the shape that used to reach the guest one call later as a
///   poisoned-lock unwrap.
pub fn run_device_call<T>(
    what: &str,
    f: impl FnOnce() -> Result<T, crate::compute::ComputeError>,
) -> Result<T, crate::compute::ComputeError> {
    match catch_mut(false, f) {
        Err(text) => Err(crate::compute::ComputeError::Device(format!(
            "{what} panicked on the device lane: {text}"
        ))),
        Ok(inner) => match take_report() {
            None => inner,
            Some(text) => Err(crate::compute::ComputeError::Device(format!(
                "{what}: a device task failed and its stack discarded the error; the recorded \
                 panic was: {text}"
            ))),
        },
    }
}
