//! `daemon-transport` — wire transport for cross-cut management/§17 traffic (deferred).
//!
//! When a placement cut puts a child host in another process/node, the management protocol runs over
//! this transport instead of in-process. Gated behind the `remote` feature; a deferred stub until
//! distribution work begins. Depends on `daemon-common` + `daemon-protocol`.

#![forbid(unsafe_code)]

// TODO (deferred): remote transport for cross-cut management/§17 framing.
