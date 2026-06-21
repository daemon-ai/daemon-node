//! Browser supervisor errors.

/// What went wrong driving the browser.
#[derive(Debug, thiserror::Error)]
pub enum BrowserError {
    /// No Chromium/Chrome could be launched (not installed, bad path, or launch timeout).
    #[error("browser launch failed: {0}")]
    Launch(String),
    /// A CDP command/transport fault during an operation (the session is torn down + respawned).
    #[error("browser operation failed: {0}")]
    Cdp(String),
    /// A filesystem error (e.g. writing a screenshot).
    #[error("browser io error: {0}")]
    Io(String),
    /// The crash-loop breaker tripped: too many consecutive launch failures.
    #[error("browser unavailable (crash-loop breaker tripped after {0} failed launches)")]
    CrashLoop(u32),
}
