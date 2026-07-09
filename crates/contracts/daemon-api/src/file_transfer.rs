// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! File-transfer **DTO behavior logic** ported from libpurple (work package W2-H).
//!
//! This module adds constructors, a state machine, and an in-memory manager over the wire
//! [`FileTransfer`] DTO in [`crate`]:
//!
//! - [`FileTransfer::new_send`] / [`FileTransfer::new_receive`] ← `purple_file_transfer_new_send` /
//!   `purple_file_transfer_new_receive` (`purplefiletransfer.c`).
//! - the state machine ([`FileTransfer::set_state`], [`FileTransfer::fail`],
//!   [`FileTransfer::advance`], [`FileTransfer::record_progress`], predicates) ←
//!   `PurpleFileTransferState` (`purplefiletransfer.h`) + `set_state`/`set_error`.
//! - [`FileTransferManager`] ← `PurpleFileTransferManager` (`purplefiletransfermanager.c`).
//!
//! The `account`/`local-file`/`cancellable` GObject properties are intentionally not ported (the
//! daemon has no `PurpleAccount`/`GFile`/`GCancellable`; see `docs/port-ledger/filetransfer.md`).

use crate::{ContactInfo, FileTransfer, FileTransferDirection, FileTransferState};
use daemon_common::{BlobRef, ContentHash};

impl Default for FileTransfer {
    fn default() -> Self {
        Self {
            name: String::new(),
            blob: BlobRef::new(ContentHash::new([0u8; 32]), 0),
            direction: FileTransferDirection::default(),
            state: FileTransferState::default(),
            remote: None,
            initiator: None,
            file_size: 0,
            transferred: 0,
            content_type: None,
            message: None,
            error: None,
            source: None,
        }
    }
}

/// Why a guarded [`FileTransfer::advance`] transition was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileTransferTransitionError {
    /// The requested state is not reachable from the current one.
    IllegalTransition {
        /// The current state.
        from: FileTransferState,
        /// The rejected target state.
        to: FileTransferState,
    },
}

impl FileTransferState {
    /// Whether this is a terminal state (`Finished` or `Failed`).
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            FileTransferState::Finished | FileTransferState::Failed
        )
    }
}

impl FileTransfer {
    /// Build an outbound transfer (← `purple_file_transfer_new_send`): `direction = Send`,
    /// `initiator = account` (the sending side), `remote = remote`, with `name`/`file_size`
    /// derived from the content-addressed `blob` (the daemon analogue of libpurple reading the
    /// `GFile`'s display name + size).
    pub fn new_send(account: ContactInfo, remote: ContactInfo, blob: BlobRef) -> Self {
        let name = blob.name.clone().unwrap_or_default();
        let file_size = blob.size;
        Self {
            name,
            blob,
            direction: FileTransferDirection::Send,
            state: FileTransferState::Unknown,
            remote: Some(remote),
            initiator: Some(account),
            file_size,
            transferred: 0,
            content_type: None,
            message: None,
            error: None,
            source: None,
        }
    }

    /// Build an inbound transfer (← `purple_file_transfer_new_receive`): `direction = Receive`,
    /// `initiator = remote` (the sending side is the remote), `remote = remote`. `blob` is a
    /// placeholder destination handle (a zero hash of the advertised size) until the bytes are
    /// stored on `receive`.
    pub fn new_receive(remote: ContactInfo, name: String, file_size: u64) -> Self {
        Self {
            name,
            blob: BlobRef::new(daemon_common::ContentHash::new([0u8; 32]), file_size),
            direction: FileTransferDirection::Receive,
            state: FileTransferState::Unknown,
            remote: Some(remote.clone()),
            initiator: Some(remote),
            file_size,
            transferred: 0,
            content_type: None,
            message: None,
            error: None,
            source: None,
        }
    }

    /// Whether this is an outbound (send) transfer.
    pub fn is_send(&self) -> bool {
        self.direction == FileTransferDirection::Send
    }

    /// Whether this is an inbound (receive) transfer.
    pub fn is_receive(&self) -> bool {
        self.direction == FileTransferDirection::Receive
    }

    /// Whether the transfer is in a terminal state (`Finished`/`Failed`).
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Whether every advertised byte has been transferred (and the size is non-zero, or a zero-byte
    /// file that reached `Finished`).
    pub fn is_complete(&self) -> bool {
        self.transferred >= self.file_size && self.state == FileTransferState::Finished
    }

    /// Set the state unconditionally (← `purple_file_transfer_set_state`; the protocol plugin owns
    /// ordering, so this is an unguarded setter).
    pub fn set_state(&mut self, state: FileTransferState) {
        self.state = state;
    }

    /// Mark the transfer failed with a reason (← `set_state(FAILED)` + `set_error`).
    pub fn fail(&mut self, error: impl Into<String>) {
        self.state = FileTransferState::Failed;
        self.error = Some(error.into());
    }

    /// The node-authoritative guarded transition: `Unknown → Negotiating → Started → Finished`, and
    /// any non-terminal state may go to `Failed`. Rejects backward/illegal transitions. (Daemon
    /// extension; libpurple's `set_state` is unguarded.)
    pub fn advance(&mut self, to: FileTransferState) -> Result<(), FileTransferTransitionError> {
        let ok = match (self.state, to) {
            (from, FileTransferState::Failed) => !from.is_terminal(),
            (FileTransferState::Unknown, FileTransferState::Negotiating) => true,
            (FileTransferState::Negotiating, FileTransferState::Started) => true,
            (FileTransferState::Started, FileTransferState::Finished) => true,
            _ => false,
        };
        if ok {
            self.state = to;
            Ok(())
        } else {
            Err(FileTransferTransitionError::IllegalTransition {
                from: self.state,
                to,
            })
        }
    }

    /// Add `n` transferred bytes, clamped to `file_size` (progress never exceeds the advertised
    /// size). Returns the new `transferred` total.
    pub fn record_progress(&mut self, n: u64) -> u64 {
        self.transferred = self.transferred.saturating_add(n).min(self.file_size);
        self.transferred
    }
}

// ---------------------------------------------------------------------------
// FileTransferManager  (← purplefiletransfermanager.c)
// ---------------------------------------------------------------------------

/// An in-memory registry of active [`FileTransfer`]s (← `PurpleFileTransferManager`, a `GListModel`
/// with `added`/`removed` + `transfer-changed[::detail]` signals). GObject signals are modeled as
/// monotonic counters: `added`/`removed` count add/remove calls; `changed_generic` counts every
/// change; `changed_state` counts state changes only (the `::state`-detailed signal).
#[derive(Debug, Default)]
pub struct FileTransferManager {
    transfers: Vec<FileTransfer>,
    added: u32,
    removed: u32,
    changed_generic: u32,
    changed_state: u32,
}

impl FileTransferManager {
    /// A new, empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of tracked transfers (← `g_list_model_get_n_items`).
    pub fn len(&self) -> usize {
        self.transfers.len()
    }

    /// Whether no transfers are tracked.
    pub fn is_empty(&self) -> bool {
        self.transfers.is_empty()
    }

    /// The transfer at `index`, if any.
    pub fn get(&self, index: usize) -> Option<&FileTransfer> {
        self.transfers.get(index)
    }

    /// How many times [`add`](Self::add) has fired (← the `added` signal count).
    pub fn added_count(&self) -> u32 {
        self.added
    }

    /// How many times [`remove`](Self::remove) has fired (← the `removed` signal count).
    pub fn removed_count(&self) -> u32 {
        self.removed
    }

    /// How many `transfer-changed` (generic) signals have fired.
    pub fn changed_count(&self) -> u32 {
        self.changed_generic
    }

    /// How many `transfer-changed::state` (detailed) signals have fired.
    pub fn changed_state_count(&self) -> u32 {
        self.changed_state
    }

    /// Add a transfer (← `purple_file_transfer_manager_add`), emitting the `added` signal.
    pub fn add(&mut self, transfer: FileTransfer) {
        self.transfers.push(transfer);
        self.added += 1;
    }

    /// Remove the transfer at `index` (← `purple_file_transfer_manager_remove`), emitting `removed`.
    /// Returns the removed transfer, or `None` if the index is out of range.
    pub fn remove(&mut self, index: usize) -> Option<FileTransfer> {
        if index >= self.transfers.len() {
            return None;
        }
        let t = self.transfers.remove(index);
        self.removed += 1;
        Some(t)
    }

    /// Change a tracked transfer's state, emitting both the generic and the `::state` change
    /// signals (← `set_state` propagating through the manager). No-op if the index is out of range.
    pub fn set_transfer_state(&mut self, index: usize, state: FileTransferState) {
        if let Some(t) = self.transfers.get_mut(index) {
            t.set_state(state);
            self.changed_generic += 1;
            self.changed_state += 1;
        }
    }

    /// Change a tracked transfer's message, emitting only the generic change signal (a non-`state`
    /// property change). No-op if the index is out of range.
    pub fn set_transfer_message(&mut self, index: usize, message: Option<String>) {
        if let Some(t) = self.transfers.get_mut(index) {
            t.message = message;
            self.changed_generic += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileTransferState;
    use daemon_common::{BlobRef, ContentHash};

    fn contact(id: &str) -> ContactInfo {
        ContactInfo {
            id: id.to_string(),
            ..ContactInfo::default()
        }
    }

    fn blob(name: &str, size: u64) -> BlobRef {
        BlobRef {
            hash: ContentHash::new([7u8; 32]),
            size,
            name: Some(name.to_string()),
            mime: None,
        }
    }

    // ---- test_file_transfer.c ----

    #[test]
    fn ft_new_send_initiator_is_account_name_and_size() {
        // ⟵ /file-transfer/new/send: initiator == the account's contact; filename == the file base
        // name; file_size == the file's size.
        let account = contact("@me:hs");
        let remote = contact("@you:hs");
        let ft = FileTransfer::new_send(account.clone(), remote.clone(), blob("cat.png", 1337));

        assert!(ft.is_send());
        assert_eq!(ft.initiator.as_ref(), Some(&account));
        assert_eq!(ft.remote.as_ref(), Some(&remote));
        assert_eq!(ft.name, "cat.png");
        assert_eq!(ft.file_size, 1337);
    }

    #[test]
    fn ft_new_receive_initiator_is_remote() {
        // ⟵ /file-transfer/new/receive: initiator == remote.
        let remote = contact("@you:hs");
        let ft = FileTransfer::new_receive(remote.clone(), "foo".to_string(), 0);

        assert!(ft.is_receive());
        assert_eq!(ft.initiator.as_ref(), Some(&remote));
        assert_eq!(ft.remote.as_ref(), Some(&remote));
        assert_eq!(ft.name, "foo");
    }

    #[test]
    fn ft_properties_roundtrip() {
        // ⟵ /file-transfer/properties: every settable property round-trips.
        let account = contact("@me:hs");
        let remote = contact("@you:hs");
        let mut ft = FileTransfer::new_send(account, remote, blob("doc.bin", 1337));
        ft.content_type = Some("application/octet-stream".to_string());
        ft.message = Some("have you heard the word?".to_string());
        ft.set_state(FileTransferState::Started);

        assert_eq!(ft.file_size, 1337);
        assert_eq!(ft.content_type.as_deref(), Some("application/octet-stream"));
        assert_eq!(ft.message.as_deref(), Some("have you heard the word?"));
        assert_eq!(ft.state, FileTransferState::Started);
    }

    // ---- state machine (+ daemon-native) ----

    #[test]
    fn ft_state_default_unknown() {
        assert_eq!(FileTransferState::default(), FileTransferState::Unknown);
    }

    #[test]
    fn ft_set_state() {
        let mut ft = FileTransfer::new_receive(contact("@r:hs"), "f".into(), 0);
        ft.set_state(FileTransferState::Negotiating);
        assert_eq!(ft.state, FileTransferState::Negotiating);
        // Unguarded: any value may be set directly (mirrors libpurple).
        ft.set_state(FileTransferState::Unknown);
        assert_eq!(ft.state, FileTransferState::Unknown);
    }

    #[test]
    fn ft_fail_sets_failed_and_error() {
        let mut ft = FileTransfer::new_receive(contact("@r:hs"), "f".into(), 0);
        ft.fail("network down");
        assert_eq!(ft.state, FileTransferState::Failed);
        assert_eq!(ft.error.as_deref(), Some("network down"));
        assert!(ft.is_terminal());
    }

    #[test]
    fn ft_advance_lifecycle_and_rejects_backwards() {
        let mut ft = FileTransfer::new_send(contact("@m:hs"), contact("@r:hs"), blob("f", 3));
        assert!(ft.advance(FileTransferState::Negotiating).is_ok());
        assert!(ft.advance(FileTransferState::Started).is_ok());
        assert!(ft.advance(FileTransferState::Finished).is_ok());
        // Finished is terminal: cannot advance further, not even to Failed.
        assert_eq!(
            ft.advance(FileTransferState::Failed),
            Err(FileTransferTransitionError::IllegalTransition {
                from: FileTransferState::Finished,
                to: FileTransferState::Failed,
            })
        );
        // A fresh transfer cannot skip straight to Started.
        let mut ft2 = FileTransfer::new_send(contact("@m:hs"), contact("@r:hs"), blob("f", 3));
        assert!(ft2.advance(FileTransferState::Started).is_err());
        // Any non-terminal state may fail.
        assert!(ft2.advance(FileTransferState::Failed).is_ok());
    }

    #[test]
    fn ft_record_progress_clamps() {
        let mut ft = FileTransfer::new_receive(contact("@r:hs"), "f".into(), 10);
        assert_eq!(ft.record_progress(4), 4);
        assert_eq!(ft.record_progress(4), 8);
        // Clamped at file_size.
        assert_eq!(ft.record_progress(100), 10);
    }

    #[test]
    fn ft_predicates() {
        let mut ft = FileTransfer::new_send(contact("@m:hs"), contact("@r:hs"), blob("f", 5));
        assert!(ft.is_send() && !ft.is_receive());
        assert!(!ft.is_terminal() && !ft.is_complete());
        ft.record_progress(5);
        ft.set_state(FileTransferState::Finished);
        assert!(ft.is_complete());
        assert!(ft.is_terminal());
    }

    // ---- test_file_transfer_manager.c ----

    #[test]
    fn manager_add_remove_emits_signals() {
        // ⟵ /file-transfer-manager/add-remove.
        let mut mgr = FileTransferManager::new();
        assert_eq!(mgr.len(), 0);

        mgr.add(FileTransfer::new_receive(
            contact("@r:hs"),
            "foo.bar".into(),
            0,
        ));
        assert_eq!(mgr.len(), 1);
        assert_eq!(mgr.added_count(), 1);
        assert_eq!(mgr.removed_count(), 0);

        mgr.remove(0);
        assert_eq!(mgr.len(), 0);
        assert_eq!(mgr.added_count(), 1);
        assert_eq!(mgr.removed_count(), 1);
    }

    #[test]
    fn manager_propagates_notify() {
        // ⟵ /file-transfer-manager/propagates-notify: a state change bumps generic + ::state; a
        // message change bumps only generic.
        let mut mgr = FileTransferManager::new();
        mgr.add(FileTransfer::new_receive(
            contact("@r:hs"),
            "foo.bar".into(),
            0,
        ));
        assert_eq!(mgr.changed_count(), 0);
        assert_eq!(mgr.changed_state_count(), 0);

        mgr.set_transfer_state(0, FileTransferState::Negotiating);
        assert_eq!(mgr.changed_count(), 1);
        assert_eq!(mgr.changed_state_count(), 1);

        mgr.set_transfer_message(0, Some("heyo".to_string()));
        assert_eq!(mgr.changed_count(), 2);
        assert_eq!(mgr.changed_state_count(), 1);
    }
}
