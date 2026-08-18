// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The in-process outbound delivery seams: the live [`DeliveryHost`] push-sink registry
//! (daemon-event-io-spec §5.9.3) and the [`CronDelivery`](crate::CronDelivery) directive resolver
//! that posts a scheduled run's text onto live session targets.

use super::*;

/// The **in-process** outbound push-registration surface (daemon-event-io-spec §5.9.3): an embedder
/// that holds the live [`NodeApiImpl`] hands the host a [`DeliverySink`] keyed by transport instance
/// so the per-session pump pushes outbound entries straight to it. This is deliberately *not* part
/// of the wire [`daemon_api::NodeApi`] surface — a sink is a live trait object that cannot cross a
/// process boundary, so cross-process transports use the pull path (`delivery_sessions` +
/// `subscribe`) instead. Registration is a live handle, not a wire op.
pub trait DeliveryHost: Send + Sync {
    /// Register (or replace) the push sink for `transport`; takes effect on the next pumped event.
    fn register_delivery_sink(&self, transport: TransportId, sink: Arc<dyn DeliverySink>);
    /// Drop the push sink for `transport` (its sessions revert to pull-only delivery).
    fn unregister_delivery_sink(&self, transport: &TransportId);
}

impl DeliveryHost for NodeApiImpl {
    fn register_delivery_sink(&self, transport: TransportId, sink: Arc<dyn DeliverySink>) {
        // One registration covers both rails: live residencies (Foreign, legacy) and the durable
        // attachment hubs (§8 — durable sessions push through their hub's pump).
        if let Some(hubs) = &self.attachments {
            hubs.register_delivery_sink(transport.clone(), sink.clone());
        }
        self.live.register_delivery_sink(transport, sink);
    }

    fn unregister_delivery_sink(&self, transport: &TransportId) {
        if let Some(hubs) = &self.attachments {
            hubs.unregister_delivery_sink(transport);
        }
        self.live.unregister_delivery_sink(transport);
    }
}

impl NodeApiImpl {
    /// Resolve a cron `deliver` directive to its concrete [`DeliveryTarget`]s, reusing the same
    /// origin/routing surface a live submit uses: `"origin"` is the job's captured origin's
    /// `primary_target()` (empty when no origin was captured — store-only fallback); `"all"` is every
    /// live session's `Primary` target (broadcast to active conversations); anything else is parsed as
    /// an explicit `"<transport>:<chat>"` direct target (split on the first `:`).
    fn resolve_delivery(&self, deliver: &str, origin: Option<&Origin>) -> Vec<DeliveryTarget> {
        match deliver.trim() {
            "origin" => origin.map(|o| vec![o.primary_target()]).unwrap_or_default(),
            // §8 sidecar re-homing: broadcast unions the live residencies' targets with the
            // attachment hubs' (durable sessions), deduped by (transport, route).
            "all" => {
                let mut out = self.live.all_primary_targets();
                if let Some(hubs) = &self.attachments {
                    for t in hubs.all_primary_targets() {
                        if !out
                            .iter()
                            .any(|o| o.transport == t.transport && o.route == t.route)
                        {
                            out.push(t);
                        }
                    }
                }
                out
            }
            spec => match spec.split_once(':') {
                Some((transport, route)) if !transport.is_empty() && !route.is_empty() => {
                    vec![DeliveryTarget::new(transport, route, SinkKind::Primary)]
                }
                _ => Vec::new(),
            },
        }
    }
}

#[async_trait]
impl crate::CronDelivery for NodeApiImpl {
    async fn deliver(&self, deliver: &str, origin: Option<&Origin>, text: &str) {
        for target in self.resolve_delivery(deliver, origin) {
            // Attribute the delivered entry to the job's creating origin when known, else to a
            // host-internal origin on the target transport (a scheduled, principal-less push).
            let entry_origin = origin
                .cloned()
                .unwrap_or_else(|| Origin::internal(target.transport.clone()));
            // Carry the run's result as a single assistant text delta — the same outbound shape a
            // live reply takes, so a registered sink projects it to a message unchanged.
            let entry = SessionLogEntry::new(
                0,
                entry_origin,
                SessionPayload::Event(AgentEvent::TextDelta {
                    seq: 0,
                    text: text.to_string(),
                }),
            );
            // Both rails share the sink registration; deliver through whichever holds the sink
            // (the sink maps are kept in lockstep by `register_delivery_sink`, so pushing via
            // one is sufficient — prefer the durable registry when armed).
            match &self.attachments {
                Some(hubs) => hubs.push_to_target(target, entry).await,
                None => self.live.push_to_target(target, entry).await,
            }
        }
    }
}
