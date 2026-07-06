// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The central ingress governor (OpenClaw Cluster F, Phase 4).
//!
//! Every network-facing carrier for the node surface — the length-framed Unix-socket / named-pipe /
//! TLS-TCP / cross-node `remote` transports and the WebSocket carrier — funnels its ingress
//! decisions through ONE [`IngressGovernor`] with ONE explicit [`IngressLimits`] policy, instead of
//! the pre-Phase-4 scatter (a frame cap duplicated across three read paths, an implicit-only decoded
//! bound, and no rate/concurrency limits at all). This collapses the limits into a single, testable,
//! fail-closed choke point and gives the "new limit = build break" (no wildcard arm) discipline one
//! home: [`IngressReject`] is exhaustively matched everywhere.
//!
//! Four limits, all fail-closed:
//! * **max frame bytes** — reject an oversize length prefix *before* the receive buffer is allocated
//!   (the pre-alloc DoS guard, unifying the scattered `MAX_FRAME_BYTES` checks).
//! * **max decoded bytes** — the post-decode *payload* cap: an O(1) check of a decoded request's
//!   carried byte payload (blob puts) at the request boundary, catching a payload that is large
//!   relative to its framing (or, for a future compressed transport, expands on decode). This is a
//!   distinct, application-level bound from the whole-message frame cap above (which, for the WS
//!   carrier, is what tungstenite's post-inflate `max_message_size` enforces).
//! * **per-peer connection rate** — a token bucket per peer IP over *new* connections, so one source
//!   cannot flood the accept loop.
//! * **connection concurrency** — a global semaphore bounding live networked connections, so a
//!   connection flood cannot exhaust memory / file descriptors unbounded.
//!
//! **Local trust is exempt from rate + concurrency.** The Unix socket / Windows named pipe are the
//! deliberate trusted local admin/FFI/CLI path and carry a [`PeerKey::Local`] sentinel: they still
//! get the frame/decoded caps but never consume the per-peer buckets or the connection budget, so a
//! *networked* connection flood can never starve the local operator CLI of a slot.
//!
//! The pure policy ([`IngressLimits`], [`IngressReject`], [`PeerKey`], [`RateSpec`]) is always
//! compiled; the runtime enforcer ([`IngressGovernor`], which needs `tokio::sync::Semaphore`) is
//! gated behind the `governor` feature, mirroring the `process`-gated helper on [`crate::env_policy`]
//! so non-networked consumers of this contracts crate stay runtime-free.

use std::net::IpAddr;

/// The maximum accepted length-framed frame size — the pre-alloc default, re-exported cap.
use crate::limits::MAX_FRAME_BYTES;

/// A per-peer token-bucket rate: `burst` tokens of capacity, refilled at `refill_per_sec` tokens per
/// second. One token is consumed per governed event (a new connection).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RateSpec {
    /// Bucket capacity — the largest instantaneous burst allowed.
    pub burst: f64,
    /// Steady-state refill rate, tokens per second.
    pub refill_per_sec: f64,
}

/// The peer a governed connection is attributed to. Networked carriers key by the peer **IP**
/// (never the ephemeral port, so a rapidly-reconnecting client is still throttled); the local-trust
/// carriers use [`PeerKey::Local`], which the governor exempts from rate + concurrency.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PeerKey {
    /// A local-trust carrier (Unix socket / Windows named pipe): exempt from rate + concurrency.
    Local,
    /// A networked peer, keyed by source IP.
    Ip(IpAddr),
}

impl PeerKey {
    /// The networked key for a peer socket address (drops the ephemeral port).
    pub fn ip(addr: IpAddr) -> Self {
        PeerKey::Ip(addr)
    }
}

/// Why the governor refused an ingress event. Exhaustive by design (**no** catch-all / `_` arm at any
/// match site): adding a limit variant must break every decision site so a new limit can never be
/// silently permitted. Every variant is a *deny*.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum IngressReject {
    /// The declared frame length exceeds [`IngressLimits::max_frame_bytes`] (rejected pre-alloc).
    #[error("ingress frame too large: {len} > {max} bytes")]
    FrameTooLarge {
        /// The declared/observed length.
        len: usize,
        /// The configured maximum.
        max: usize,
    },
    /// The decoded payload exceeds [`IngressLimits::max_decoded_bytes`].
    #[error("ingress decoded payload too large: {len} > {max} bytes")]
    DecodedTooLarge {
        /// The decoded payload length.
        len: usize,
        /// The configured maximum.
        max: usize,
    },
    /// The live-connection concurrency ceiling ([`IngressLimits::max_connections`]) is reached.
    #[error("ingress connection cap reached: {max} live connections")]
    ConnectionCapReached {
        /// The configured maximum.
        max: usize,
    },
    /// The peer's new-connection token bucket is empty (per-peer rate exceeded).
    #[error("ingress per-peer connection rate exceeded")]
    PeerRateExceeded,
}

impl IngressReject {
    /// Map a reject to the `InvalidData` I/O error the length-framed read paths surface (so an
    /// oversize frame closes the connection exactly as the pre-governor inline check did).
    pub fn as_io_error(&self) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, self.to_string())
    }
}

/// The single ingress policy every carrier enforces. Pure data (`Copy`); the stateful enforcement
/// lives in [`IngressGovernor`]. Construct via [`IngressLimits::default`] (the secure-by-default
/// generous posture) or [`IngressLimits::unlimited`] (caps off — client/test read paths).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IngressLimits {
    /// Max accepted length-framed wire frame, rejected *before* the receive buffer is allocated.
    pub max_frame_bytes: usize,
    /// Max post-decode payload size (post-decompression for a compressed carrier; the decoded
    /// byte-payload of a request for the blob-carrying variants).
    pub max_decoded_bytes: usize,
    /// Max concurrent live connections across the governed (networked) carriers; `None` = unbounded.
    pub max_connections: Option<usize>,
    /// Per-peer token-bucket rate for *new* connections; `None` = no per-peer rate limit.
    pub peer_conn_rate: Option<RateSpec>,
    /// Upper bound on distinct peers the rate limiter tracks (its own memory-DoS guard); beyond it,
    /// new peers share a single overflow bucket.
    pub max_tracked_peers: usize,
}

impl Default for IngressLimits {
    /// The secure-by-default posture (see the module + `HARDENING-PLAN.md`): generous enough that no
    /// legitimate frame/blob/burst is newly rejected, tight enough to bound an unbounded flood.
    ///
    /// * `max_frame_bytes` / `max_decoded_bytes` = [`MAX_FRAME_BYTES`] (640 MiB) — the existing
    ///   pre-alloc ceiling; ≥ the 256 MiB blob ceiling so no in-spec `BlobPut` is newly rejected.
    /// * `max_connections` = 1024 — a high networked-concurrency ceiling.
    /// * `peer_conn_rate` = burst 256, refill 128/s — well above any real client's connection burst.
    /// * `max_tracked_peers` = 4096.
    fn default() -> Self {
        Self {
            max_frame_bytes: MAX_FRAME_BYTES,
            max_decoded_bytes: MAX_FRAME_BYTES,
            max_connections: Some(1024),
            peer_conn_rate: Some(RateSpec {
                burst: 256.0,
                refill_per_sec: 128.0,
            }),
            max_tracked_peers: 4096,
        }
    }
}

impl IngressLimits {
    /// All caps off (frame/decoded at `usize::MAX`, no concurrency or rate limit). For the client
    /// read paths and pure unit tests that must never be throttled by the ingress policy.
    pub fn unlimited() -> Self {
        Self {
            max_frame_bytes: usize::MAX,
            max_decoded_bytes: usize::MAX,
            max_connections: None,
            peer_conn_rate: None,
            max_tracked_peers: 0,
        }
    }

    /// Reject an oversize declared frame length *before* any receive buffer is allocated. The single
    /// home of the pre-alloc frame cap that the length-framed read paths (socket / remote / ws) call.
    pub fn check_frame_len(&self, len: usize) -> Result<(), IngressReject> {
        if len > self.max_frame_bytes {
            Err(IngressReject::FrameTooLarge {
                len,
                max: self.max_frame_bytes,
            })
        } else {
            Ok(())
        }
    }

    /// Reject an oversize decoded payload (the post-decode / post-inflate cap). O(1): the caller
    /// passes the length of a payload already in hand (e.g. a decoded blob's `len()`), so there is no
    /// second allocation.
    pub fn check_decoded_len(&self, len: usize) -> Result<(), IngressReject> {
        if len > self.max_decoded_bytes {
            Err(IngressReject::DecodedTooLarge {
                len,
                max: self.max_decoded_bytes,
            })
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "governor")]
pub use governor_impl::{ConnectionPermit, IngressGovernor};

#[cfg(feature = "governor")]
mod governor_impl {
    use super::{IngressLimits, IngressReject, PeerKey, RateSpec};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    use tokio::sync::{OwnedSemaphorePermit, Semaphore};

    /// A single peer's token bucket over new connections. Pure arithmetic; unit-testable with
    /// synthetic `Instant`s.
    struct TokenBucket {
        tokens: f64,
        last: Instant,
    }

    impl TokenBucket {
        fn new(spec: RateSpec, now: Instant) -> Self {
            Self {
                tokens: spec.burst,
                last: now,
            }
        }

        /// Refill for the elapsed time (capped at `burst`) then take one token if available.
        fn try_take(&mut self, now: Instant, spec: RateSpec) -> bool {
            let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
            self.tokens = (self.tokens + elapsed * spec.refill_per_sec).min(spec.burst);
            self.last = now;
            if self.tokens >= 1.0 {
                self.tokens -= 1.0;
                true
            } else {
                false
            }
        }
    }

    /// The per-peer bucket table plus a single shared overflow bucket used once `max_tracked_peers`
    /// distinct peers are tracked — so the limiter's own state can never be a memory DoS.
    struct PeerTable {
        buckets: HashMap<PeerKey, TokenBucket>,
        overflow: TokenBucket,
    }

    impl PeerTable {
        fn new(spec: RateSpec, now: Instant) -> Self {
            Self {
                buckets: HashMap::new(),
                overflow: TokenBucket::new(spec, now),
            }
        }

        fn take(
            &mut self,
            key: &PeerKey,
            now: Instant,
            spec: RateSpec,
            max_tracked: usize,
        ) -> bool {
            if let Some(bucket) = self.buckets.get_mut(key) {
                return bucket.try_take(now, spec);
            }
            if self.buckets.len() < max_tracked {
                let mut bucket = TokenBucket::new(spec, now);
                let ok = bucket.try_take(now, spec);
                self.buckets.insert(key.clone(), bucket);
                ok
            } else {
                self.overflow.try_take(now, spec)
            }
        }
    }

    /// A held connection slot. Acquired non-blocking at accept and dropped (RAII) when the
    /// connection closes — including on the Cluster-F secret-epoch revocation teardown, which returns
    /// through the same close path — releasing the slot. A no-op when concurrency is unbounded.
    pub struct ConnectionPermit(#[allow(dead_code)] Option<OwnedSemaphorePermit>);

    /// The stateful ingress enforcer: the [`IngressLimits`] policy plus the connection semaphore and
    /// the per-peer buckets. Cheap to share (`Arc`); one instance is built at boot and threaded into
    /// every networked carrier.
    ///
    /// Concurrency: the `peers` `Mutex` is a strict **leaf** — held only for the O(1) bucket update
    /// in [`check_peer`](IngressGovernor::check_peer), never across an `.await`, and never nested
    /// under any other lock. [`admit_connection`](IngressGovernor::admit_connection) is non-blocking
    /// (`try_acquire`), so no acquire order can invert against the revocation teardown.
    pub struct IngressGovernor {
        limits: IngressLimits,
        conns: Option<Arc<Semaphore>>,
        peers: Mutex<PeerTable>,
    }

    impl IngressGovernor {
        /// Build a governor from an explicit policy.
        pub fn new(limits: IngressLimits) -> Arc<Self> {
            let conns = limits.max_connections.map(|n| Arc::new(Semaphore::new(n)));
            // The overflow bucket only matters when a rate is set; a placeholder spec is fine
            // otherwise (check_peer returns early when peer_conn_rate is None).
            let spec = limits.peer_conn_rate.unwrap_or(RateSpec {
                burst: 1.0,
                refill_per_sec: 1.0,
            });
            Arc::new(Self {
                limits,
                conns,
                peers: Mutex::new(PeerTable::new(spec, Instant::now())),
            })
        }

        /// The secure-by-default governor ([`IngressLimits::default`]).
        pub fn secure_default() -> Arc<Self> {
            Self::new(IngressLimits::default())
        }

        /// A governor with every cap off ([`IngressLimits::unlimited`]) — for tests / trusted paths.
        pub fn unlimited() -> Arc<Self> {
            Self::new(IngressLimits::unlimited())
        }

        /// This governor's policy (`Copy`), for the frame/decoded read-path checks.
        pub fn limits(&self) -> IngressLimits {
            self.limits
        }

        /// Reject an oversize frame length pre-alloc (delegates to [`IngressLimits::check_frame_len`]).
        pub fn check_frame_len(&self, len: usize) -> Result<(), IngressReject> {
            self.limits.check_frame_len(len)
        }

        /// Reject an oversize decoded payload (delegates to [`IngressLimits::check_decoded_len`]).
        pub fn check_decoded_len(&self, len: usize) -> Result<(), IngressReject> {
            self.limits.check_decoded_len(len)
        }

        /// Try to admit one live connection: `Some(permit)` on success (held for the connection's
        /// life, RAII-released on close), `None` when the concurrency cap is reached (**fail closed**
        /// — the caller drops the connection; the accept loop is never queued/blocked). Always
        /// `Some` when concurrency is unbounded.
        pub fn admit_connection(&self) -> Option<ConnectionPermit> {
            match &self.conns {
                None => Some(ConnectionPermit(None)),
                Some(sem) => match sem.clone().try_acquire_owned() {
                    Ok(permit) => Some(ConnectionPermit(Some(permit))),
                    Err(_) => None,
                },
            }
        }

        /// Consume one per-peer connection token. `Ok(())` if allowed; `Err(PeerRateExceeded)` when
        /// the bucket is empty (**fail closed**). [`PeerKey::Local`] and a `None` rate are exempt
        /// (return `Ok`).
        pub fn check_peer(&self, key: &PeerKey) -> Result<(), IngressReject> {
            if matches!(key, PeerKey::Local) {
                return Ok(());
            }
            let Some(spec) = self.limits.peer_conn_rate else {
                return Ok(());
            };
            let now = Instant::now();
            let mut table = self.peers.lock().unwrap_or_else(|e| e.into_inner());
            if table.take(key, now, spec, self.limits.max_tracked_peers) {
                Ok(())
            } else {
                Err(IngressReject::PeerRateExceeded)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::net::{IpAddr, Ipv4Addr};
        use std::time::Duration;

        fn ip(n: u8) -> PeerKey {
            PeerKey::Ip(IpAddr::V4(Ipv4Addr::new(127, 0, 0, n)))
        }

        /// The token bucket allows `burst` immediate takes, then refuses until it refills — the
        /// per-peer rate-exceeded → refused behavior, deterministic via synthetic instants.
        #[test]
        fn token_bucket_bursts_then_refuses_then_refills() {
            let spec = RateSpec {
                burst: 3.0,
                refill_per_sec: 1.0,
            };
            let t0 = Instant::now();
            let mut b = TokenBucket::new(spec, t0);
            assert!(b.try_take(t0, spec), "1st within burst");
            assert!(b.try_take(t0, spec), "2nd within burst");
            assert!(b.try_take(t0, spec), "3rd within burst");
            assert!(
                !b.try_take(t0, spec),
                "4th over burst is refused (fail closed)"
            );
            // One second later, one token has refilled.
            let t1 = t0 + Duration::from_secs(1);
            assert!(b.try_take(t1, spec), "a refilled token is available");
            assert!(!b.try_take(t1, spec), "but only one");
        }

        /// The connection semaphore admits exactly `max_connections`, refuses beyond it (fail
        /// closed), and frees a slot when a permit drops.
        #[tokio::test]
        async fn concurrency_cap_admits_then_refuses_then_frees() {
            let gov = IngressGovernor::new(IngressLimits {
                max_connections: Some(2),
                ..IngressLimits::unlimited()
            });
            let p1 = gov.admit_connection().expect("1st admitted");
            let _p2 = gov.admit_connection().expect("2nd admitted");
            assert!(
                gov.admit_connection().is_none(),
                "3rd refused at the cap (fail closed)"
            );
            drop(p1);
            assert!(
                gov.admit_connection().is_some(),
                "dropping a permit frees a slot"
            );
        }

        /// Unbounded concurrency always admits (a no-op permit).
        #[tokio::test]
        async fn unbounded_concurrency_always_admits() {
            let gov = IngressGovernor::unlimited();
            for _ in 0..1000 {
                assert!(gov.admit_connection().is_some());
            }
        }

        /// Per-peer rate: each peer gets its own burst; the cap is per-peer, not global.
        #[tokio::test]
        async fn per_peer_rate_is_isolated_and_fails_closed() {
            let gov = IngressGovernor::new(IngressLimits {
                peer_conn_rate: Some(RateSpec {
                    burst: 2.0,
                    refill_per_sec: 0.0,
                }),
                max_tracked_peers: 16,
                ..IngressLimits::unlimited()
            });
            assert!(gov.check_peer(&ip(1)).is_ok(), "peer1 1st");
            assert!(gov.check_peer(&ip(1)).is_ok(), "peer1 2nd");
            assert_eq!(
                gov.check_peer(&ip(1)),
                Err(IngressReject::PeerRateExceeded),
                "peer1 3rd refused (fail closed)"
            );
            // A different peer is unaffected (per-peer, not global).
            assert!(gov.check_peer(&ip(2)).is_ok(), "peer2 unaffected");
        }

        /// Local trust is exempt from the per-peer rate even when the bucket for a networked peer
        /// would be exhausted.
        #[tokio::test]
        async fn local_trust_is_rate_exempt() {
            let gov = IngressGovernor::new(IngressLimits {
                peer_conn_rate: Some(RateSpec {
                    burst: 1.0,
                    refill_per_sec: 0.0,
                }),
                max_connections: Some(1),
                max_tracked_peers: 16,
                ..IngressLimits::unlimited()
            });
            // Exhaust a networked peer's single token.
            assert!(gov.check_peer(&ip(9)).is_ok());
            assert_eq!(gov.check_peer(&ip(9)), Err(IngressReject::PeerRateExceeded));
            // Local trust is never rate-limited, regardless.
            for _ in 0..100 {
                assert!(gov.check_peer(&PeerKey::Local).is_ok());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame cap fails closed above the limit and passes at/below it.
    #[test]
    fn frame_len_check_fails_closed() {
        let limits = IngressLimits {
            max_frame_bytes: 100,
            ..IngressLimits::unlimited()
        };
        assert!(limits.check_frame_len(100).is_ok(), "at the cap is allowed");
        assert!(limits.check_frame_len(0).is_ok());
        assert_eq!(
            limits.check_frame_len(101),
            Err(IngressReject::FrameTooLarge { len: 101, max: 100 }),
            "over the cap is rejected"
        );
    }

    /// The decoded cap fails closed above the limit and passes at/below it (the oversize-decoded
    /// payload rejection, measured O(1) from a length already in hand).
    #[test]
    fn decoded_len_check_fails_closed() {
        let limits = IngressLimits {
            max_decoded_bytes: 256,
            ..IngressLimits::unlimited()
        };
        assert!(limits.check_decoded_len(256).is_ok());
        assert_eq!(
            limits.check_decoded_len(257),
            Err(IngressReject::DecodedTooLarge { len: 257, max: 256 }),
            "an oversize decoded payload is rejected"
        );
    }

    /// The secure default is generous enough to never newly reject an in-spec frame/blob: the frame
    /// and decoded caps are the existing 640 MiB ceiling, comfortably above the 256 MiB blob ceiling.
    #[test]
    fn secure_default_is_generous() {
        let d = IngressLimits::default();
        assert_eq!(d.max_frame_bytes, MAX_FRAME_BYTES);
        assert_eq!(d.max_decoded_bytes, MAX_FRAME_BYTES);
        assert!(d.max_decoded_bytes >= 256 * 1024 * 1024, "≥ MAX_BLOB_SIZE");
        assert_eq!(d.max_connections, Some(1024));
        assert!(d.peer_conn_rate.is_some());
        // A max-size in-spec blob frame is accepted by both caps.
        assert!(d.check_frame_len(256 * 1024 * 1024).is_ok());
        assert!(d.check_decoded_len(256 * 1024 * 1024).is_ok());
    }

    /// `unlimited` disables every cap.
    #[test]
    fn unlimited_disables_caps() {
        let u = IngressLimits::unlimited();
        assert!(u.check_frame_len(usize::MAX).is_ok());
        assert!(u.check_decoded_len(usize::MAX).is_ok());
        assert!(u.max_connections.is_none());
        assert!(u.peer_conn_rate.is_none());
    }
}
