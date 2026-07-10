// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The verifiable-journal read/write surface: sealing management mutations onto the
//! `node-management` stream, and the shared cursor-paged history reader (with per-segment
//! verification) behind `session_history` / `conv_history` / `unit_history`.

use super::*;

impl NodeApiImpl {
    /// Journal + seal one management mutation onto the verifiable `node-management` stream (a sealed
    /// dCBOR entry per mutating `conv_*`/`member_*` op). No-op when journaling is disabled.
    pub(crate) async fn audit_management(&self, kind: &str, detail: String) {
        // Reuse one long-lived sink so the chain links per op (each `seal` advances to the next
        // segment); only build it on the first mutation, and only when journaling is enabled.
        let sink = {
            let mut guard = self.mgmt_journal.lock().unwrap();
            if guard.is_none() {
                let Some(signer) = self.verifier.clone() else {
                    return;
                };
                *guard = Some(Arc::new(JournalSink::new(
                    self.store.clone(),
                    signer,
                    JournalStreamId::unit(&UnitId::new("node-management")),
                )));
            }
            guard.as_ref().unwrap().clone()
        };
        if let Err(e) = sink.record_management(kind.to_string(), detail).await {
            tracing::warn!(error = %e, kind, "management audit: record failed");
            return;
        }
        if let Err(e) = sink.seal().await {
            tracing::warn!(error = %e, kind, "management audit: seal failed");
        }
    }

    /// Run a management operation, then record one audit entry (op-then-audit, audit only on
    /// success). Centralizes the `op.await?; audit_management(..); Ok(..)` shape shared by the
    /// conv/member/contact mutating wrappers.
    pub(crate) async fn audited<T>(
        &self,
        kind: &str,
        detail: String,
        op: impl std::future::Future<Output = Result<T, ApiError>>,
    ) -> Result<T, ApiError> {
        let out = op.await?;
        self.audit_management(kind, detail).await;
        Ok(out)
    }

    /// Journal + seal one conversation chat message onto the verifiable `conv:<transport>:<conv>`
    /// stream (wire v38) — the append half of the [`LifecycleSink::chat_message`] seam every
    /// messaging adapter reports its sends/deliveries through. One long-lived sink per stream
    /// (mirroring [`Self::audit_management`]'s `mgmt_journal`) so the chain links per message and
    /// concurrent appends to one conversation share a segment/seq source. Returns whether the
    /// record landed durably (the caller emits `MessagesChanged` only then); no-op `false` when
    /// journaling is disabled.
    pub(crate) async fn journal_chat_message(
        &self,
        transport: &TransportId,
        conv: &str,
        message: &daemon_api::ChatMessage,
        origin_op: Option<String>,
    ) -> bool {
        let stream = JournalStreamId::unit(&UnitId::new(format!(
            "conv:{}:{}",
            transport.as_str(),
            conv
        )));
        let sink = {
            let mut guard = self.chat_journals.lock().unwrap();
            match guard.get(&stream) {
                Some(sink) => sink.clone(),
                None => {
                    let Some(signer) = self.verifier.clone() else {
                        return false;
                    };
                    let sink =
                        Arc::new(JournalSink::new(self.store.clone(), signer, stream.clone()));
                    guard.insert(stream, sink.clone());
                    sink
                }
            }
        };
        if let Err(e) = sink.record_chat(message, origin_op).await {
            tracing::warn!(error = %e, transport = %transport.as_str(), conv, "chat journal: record failed");
            return false;
        }
        // Seal per message so each record verifies immediately (`JournalRecord::verified`); a seal
        // hiccup leaves the record durable in an open segment, so the pointer still goes out.
        if let Err(e) = sink.seal().await {
            tracing::warn!(error = %e, transport = %transport.as_str(), conv, "chat journal: seal failed");
        }
        true
    }

    /// Read a stream's durable verifiable history: cursor-page the store, decode each entry to its
    /// typed view, decode block bodies into `TranscriptBlock`s, and stamp each entry with the
    /// verification result of its sealed segment. Non-destructive (the live drains are separate).
    pub(crate) async fn read_history(
        &self,
        stream: JournalStreamId,
        after_cursor: u64,
        max: u32,
    ) -> JournalPageView {
        // Clamp the page size at this (shared SessionHistory/UnitHistory/ConvHistory handler)
        // seam: `max == 0` previously returned the ENTIRE journal, which the fixed-buffer client
        // codec cannot decode past WIRE_PAGE_MAX entries. The store's `if max > 0 { truncate }`
        // contract stays generic; `next_cursor`/`head_cursor` let the client loop to completion.
        let max = daemon_api::clamp_page_max(max);
        let page = self.store.load_journal(&stream, after_cursor, max).await;
        self.view_journal_page(stream, page).await
    }

    /// Read a stream's newest-anchored backward window (rung 2): the `max` newest entries with
    /// `cursor < before_cursor`, decoded + verified exactly like [`Self::read_history`]. The view's
    /// `next_cursor` is the backward continuation (the OLDEST returned cursor, or `before_cursor`
    /// when the window is empty); anchoring is stable because appends land above every served
    /// anchor, so interleaved writes never skip or duplicate entries across a backward walk.
    pub(crate) async fn read_history_before(
        &self,
        stream: JournalStreamId,
        before_cursor: u64,
        max: u32,
    ) -> JournalPageView {
        // Same wire-bound clamp as the forward read (`max == 0` = one full wire page, newest).
        let max = daemon_api::clamp_page_max(max);
        let page = self
            .store
            .load_journal_before(&stream, before_cursor, max)
            .await;
        self.view_journal_page(stream, page).await
    }

    /// Decode + verify one fetched journal page into its wire view (shared by the forward and
    /// backward reads; the page's cursors pass through untouched).
    async fn view_journal_page(
        &self,
        stream: JournalStreamId,
        page: daemon_store::JournalPage,
    ) -> JournalPageView {
        let key = self.verifier.as_ref().map(|s| s.verifying_key());

        // Verify each distinct sealed segment the page touches exactly once.
        let mut seg_verified: HashMap<u64, bool> = HashMap::new();
        for je in &page.entries {
            if let std::collections::hash_map::Entry::Vacant(slot) = seg_verified.entry(je.segment)
            {
                let ok = match &key {
                    Some(k) => self.verify_segment_in_store(&stream, je.segment, k).await,
                    None => false,
                };
                slot.insert(ok);
            }
        }

        let entries = page
            .entries
            .into_iter()
            .filter_map(|je| {
                let view = decode_entry(&je.entry.bytes).ok()?;
                let payload = match view.payload {
                    JournalPayload::Management { detail } => {
                        JournalRecordPayload::Management { detail }
                    }
                    JournalPayload::Block { body } => {
                        let block: TranscriptBlock = ciborium::from_reader(&body[..]).ok()?;
                        JournalRecordPayload::Block { block }
                    }
                    JournalPayload::Chat { body } => {
                        let message: daemon_api::ChatMessage =
                            ciborium::from_reader(&body[..]).ok()?;
                        JournalRecordPayload::Chat {
                            message: Box::new(message),
                        }
                    }
                };
                Some(JournalRecord {
                    cursor: je.cursor,
                    segment: je.segment,
                    seq: view.seq,
                    epoch: view.epoch,
                    trace: view.trace,
                    kind: view.kind,
                    timestamp_ms: view.timestamp_ms,
                    verified: seg_verified.get(&je.segment).copied().unwrap_or(false),
                    // rung 3 (api vNEXT): the node-owned envelope's uniform operation provenance.
                    origin_op: view.origin_op,
                    payload,
                })
            })
            .collect();

        let sealed_after = self
            .store
            .active_journal_seal(&stream)
            .await
            .map(|seal| seal.seal_cursor);

        JournalPageView {
            entries,
            next_cursor: page.next_cursor,
            head_cursor: page.head_cursor,
            sealed_after,
        }
    }

    /// Verify one sealed `(stream, segment)` against the node's verifying key: load the full
    /// segment, fold its entries onto the prior segment's sealed root, and check the signature. An
    /// open (unsealed) segment — or a broken prior link — reports `false`.
    async fn verify_segment_in_store(
        &self,
        stream: &JournalStreamId,
        segment: u64,
        key: &VerifyingKey,
    ) -> bool {
        let Some(seg) = self.store.load_trace_segment(stream, segment).await else {
            return false;
        };
        let Some(committed) = seg.committed else {
            return false;
        };
        let prior = if segment == 0 {
            GENESIS_ROOT
        } else {
            match self
                .store
                .load_trace_segment(stream, segment - 1)
                .await
                .and_then(|s| s.committed)
            {
                Some(c) => c.root,
                None => return false,
            }
        };
        let entries: Vec<(u64, Vec<u8>, ContentHash)> = seg
            .entries
            .into_iter()
            .map(|e| (e.seq, e.bytes, e.content_hash))
            .collect();
        let input = SegmentInput {
            stream,
            segment,
            prior,
            entries: &entries,
        };
        verify_segment(&input, &committed.root, &committed.signature, key).is_ok()
    }
}
