//! `daemon-provision` — workspace + placement provisioning.
//!
//! Creates the execution environment a unit runs in (working dirs, process or container sandboxes).
//! `process` (default) and `container` features select isolation backends — placement is a *cut* in
//! the unit tree, realized here. Depends only on `daemon-common`.

#![forbid(unsafe_code)]

// TODO: define Provisioner trait + process/container backends.
