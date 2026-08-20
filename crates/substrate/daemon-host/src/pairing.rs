// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The pairing arming state + SPAKE2 exchange driver (pairing spec §5.3–§5.4).
//!
//! A [`PairingManager`] holds the node's **armed** pairing state: at most one active code at a
//! time, derived into a SPAKE2 password scalar the moment it is armed (the plaintext code is
//! returned to the arming admin once and never retained). Everything here is in-memory only —
//! a node restart disarms.
//!
//! Invariants (spec §5.3, binding):
//! * TTL 120 s — expiry disarms (checked lazily on every observation).
//! * Single use — the first successful enrollment disarms.
//! * Attempt budget 5 — every non-`AuthOk` outcome of a started exchange (malformed payload,
//!   wrong code, MAC mismatch, abandoned connection) counts one attempt; exhaustion locks the
//!   manager until an admin explicitly re-arms (or cancels).
//! * One exchange in flight at a time; a concurrent second `AuthStart` is refused *without*
//!   consuming the in-flight attempt.
//!
//! The SASL mechanism side (`X-DAEMON-PAIR-1`, spec §4) lives in [`crate::authn`]; this module
//! owns the state machine and the CBOR payload codecs so both the authenticator and the admin
//! surface ([`PairingBegin`/`Status`/`Cancel`]) drive one shared object.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use daemon_pake::{Exchange, Finished, PasswordScalar, IDENT_APP, IDENT_NODE, SHARE_LEN};
use serde::{Deserialize, Serialize};

/// The SASL mechanism name (spec §4). Not an IANA-registered mechanism, hence the `X-` prefix.
pub const MECH_PAIRING: &str = "X-DAEMON-PAIR-1";

/// Armed-code time to live (spec §5.3).
pub const PAIRING_TTL: Duration = Duration::from_secs(120);

/// Failed-exchange budget before the manager locks (spec §5.3).
pub const PAIRING_ATTEMPTS: u8 = 5;

/// Pairing-code length in Crockford-base32 characters (≈50 bits from the OS CSPRNG).
pub const CODE_LEN: usize = 10;

/// Crockford base32 (no I/L/O/U), the code alphabet. Confusable folding on the *typing* side
/// maps `I`/`L`→`1` and `O`→`0`; codes generated here never contain those letters.
const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Why an exchange could not start / complete. Maps 1:1 onto the spec §4 `AuthError` reasons —
/// deliberately coarse so an unauthenticated caller learns nothing beyond the four states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairingRefusal {
    /// No armed code (never armed, expired, already used, cancelled) — `pairing-not-armed`.
    NotArmed,
    /// The connection presented no client certificate — `pairing-no-client-cert`.
    NoClientCert,
    /// The attempt budget is exhausted; an admin must re-arm — `pairing-locked`.
    Locked,
    /// Anything else: wrong code, malformed payload, MAC mismatch, concurrent exchange —
    /// `pairing-failed` (indistinguishable by design).
    Failed,
}

impl PairingRefusal {
    /// The wire `AuthError.reason` string (spec §4).
    pub fn reason(self) -> &'static str {
        match self {
            Self::NotArmed => "pairing-not-armed",
            Self::NoClientCert => "pairing-no-client-cert",
            Self::Locked => "pairing-locked",
            Self::Failed => "pairing-failed",
        }
    }
}

/// What `PairingBegin` hands back to the arming admin: the one and only exposure of the code.
#[derive(Clone, Debug)]
pub struct ArmedCode {
    /// The canonical (ungrouped, uppercase Crockford) code. Display-group as `XXXXX-XXXXX`.
    pub code: String,
    /// Wall-clock expiry, milliseconds since the Unix epoch.
    pub expires_at_ms: u64,
}

/// The `PairingStatus` view (never contains the code).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairingStatusView {
    /// Whether a live (unexpired) code is armed.
    pub armed: bool,
    /// Expiry of the armed code (ms since epoch), when armed.
    pub expires_at_ms: Option<u64>,
    /// Remaining exchange attempts before lockout, when armed.
    pub attempts_remaining: Option<u8>,
    /// Whether the attempt budget was exhausted (an admin must re-arm or cancel).
    pub locked: bool,
}

/// `AuthStart` initial payload (spec §4 message 1): the app's SPAKE2 share + display name.
#[derive(Deserialize)]
struct PairStart {
    v: u8,
    #[serde(with = "serde_bytes")]
    pa: Vec<u8>,
    name: String,
}

/// `AuthChallenge` payload (message 2): the node's share + confirmation MAC.
#[derive(Serialize)]
struct PairChallenge {
    #[serde(with = "serde_bytes")]
    pb: Vec<u8>,
    #[serde(with = "serde_bytes")]
    cb: Vec<u8>,
}

/// `AuthStep` payload (message 3): the app's confirmation MAC.
#[derive(Deserialize)]
struct PairConfirm {
    #[serde(with = "serde_bytes")]
    ca: Vec<u8>,
}

/// The armed state. The plaintext code is NOT here — only the derived scalar survives arming.
struct Armed {
    w: PasswordScalar,
    deadline: Instant,
    expires_at_ms: u64,
    attempts_remaining: u8,
    in_flight: bool,
    /// Monotonic arming generation: a completion/failure from an exchange started under an
    /// older code must not mutate the state of a newer one.
    generation: u64,
}

enum State {
    Disarmed,
    Locked,
    Armed(Box<Armed>),
}

/// The node's single, shared pairing state (spec §5.3). Cheap to share via `Arc`; the
/// authenticator drives exchanges against it and the admin API arms/cancels/inspects it.
pub struct PairingManager {
    state: Mutex<State>,
    generations: Mutex<u64>,
    /// Fired on every observable state change (arm/cancel/expire-observation/attempt/lock/
    /// enrollment), so admin UIs get the §5.5 coalescing invalidation instead of polling. Set
    /// post-assembly by the node wiring; absent in tests.
    change_hook: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

impl Default for PairingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PairingManager {
    /// A disarmed manager.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State::Disarmed),
            generations: Mutex::new(0),
            change_hook: Mutex::new(None),
        }
    }

    /// Install the state-change notifier (the §5.5 invalidation seam). One hook; a re-set
    /// replaces it.
    pub fn set_change_hook(&self, hook: Box<dyn Fn() + Send + Sync>) {
        *self.change_hook.lock().unwrap_or_else(|e| e.into_inner()) = Some(hook);
    }

    /// Fire the state-change notifier, if installed. Public so the authenticator can also signal
    /// the paired-device-set change once the §5.4 enrollment transaction lands.
    pub fn notify_changed(&self) {
        let hook = self.change_hook.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = hook.as_ref() {
            h();
        }
    }

    /// Arm a fresh code (spec §5.5 `PairingBegin`), replacing (and invalidating) any previous
    /// code and clearing a lockout. The returned [`ArmedCode`] is the only exposure of the code.
    pub fn arm(&self) -> Result<ArmedCode, PairingError> {
        let code = generate_code()?;
        let w = PasswordScalar::derive(code.as_bytes());
        let expires_at_ms = (SystemTime::now() + PAIRING_TTL)
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let generation = {
            let mut g = self.generations.lock().unwrap_or_else(|e| e.into_inner());
            *g += 1;
            *g
        };
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = State::Armed(Box::new(Armed {
            w,
            deadline: Instant::now() + PAIRING_TTL,
            expires_at_ms,
            attempts_remaining: PAIRING_ATTEMPTS,
            in_flight: false,
            generation,
        }));
        tracing::info!(
            expires_at_ms,
            "pairing: armed (ttl {}s)",
            PAIRING_TTL.as_secs()
        );
        self.notify_changed();
        Ok(ArmedCode {
            code,
            expires_at_ms,
        })
    }

    /// Disarm and clear any lockout (spec §5.5 `PairingCancel`).
    pub fn cancel(&self) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if !matches!(*state, State::Disarmed) {
                tracing::info!("pairing: disarmed by admin");
            }
            *state = State::Disarmed;
        }
        self.notify_changed();
    }

    /// The admin status view (spec §5.5 `PairingStatus`). Lazily expires a stale code.
    pub fn status(&self) -> PairingStatusView {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        expire_if_stale(&mut state);
        match &*state {
            State::Disarmed => PairingStatusView {
                armed: false,
                expires_at_ms: None,
                attempts_remaining: None,
                locked: false,
            },
            State::Locked => PairingStatusView {
                armed: false,
                expires_at_ms: None,
                attempts_remaining: None,
                locked: true,
            },
            State::Armed(a) => PairingStatusView {
                armed: true,
                expires_at_ms: Some(a.expires_at_ms),
                attempts_remaining: Some(a.attempts_remaining),
                locked: false,
            },
        }
    }

    /// Whether a live code is armed right now (gates `Hello.auth_mechanisms` advertisement).
    pub fn is_armed(&self) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        expire_if_stale(&mut state);
        matches!(*state, State::Armed(_))
    }

    /// Start one SPAKE2 exchange from the app's `AuthStart` payload (spec §4 messages 1–2).
    ///
    /// `server_fp_hex` / `client_fp_hex` are the SHA-256 leaf fingerprints as observed on THIS
    /// TLS connection (channel binding, spec §3.3). Returns the CBOR `AuthChallenge` payload and
    /// the pending exchange to complete with the app's confirmation MAC.
    pub fn begin_exchange(
        self: &Arc<Self>,
        server_fp_hex: &str,
        client_fp_hex: &str,
        initial: &[u8],
    ) -> Result<(Vec<u8>, PendingPairing), PairingRefusal> {
        // Admission control under the lock; the (cheap) EC math happens outside it.
        let (w, generation) = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            expire_if_stale(&mut state);
            let armed = match &mut *state {
                State::Disarmed => return Err(PairingRefusal::NotArmed),
                State::Locked => return Err(PairingRefusal::Locked),
                State::Armed(a) => a,
            };
            if armed.in_flight {
                // Refused WITHOUT consuming the in-flight attempt (spec §5.3).
                return Err(PairingRefusal::Failed);
            }
            armed.in_flight = true;
            (armed.w.clone(), armed.generation)
        };

        match run_spake2(&w, server_fp_hex, client_fp_hex, initial) {
            Ok((challenge, finished, display_name)) => Ok((
                challenge,
                PendingPairing {
                    manager: self.clone(),
                    finished,
                    generation,
                    client_fp: client_fp_hex.to_string(),
                    display_name,
                },
            )),
            Err(refusal) => {
                self.fail_attempt(generation, "malformed or invalid AuthStart payload");
                Err(refusal)
            }
        }
    }

    /// Record one failed exchange attempt (spec §5.3): clears in-flight and locks on exhaustion.
    /// No-op when the state has moved on (disarmed / re-armed under a newer generation).
    fn fail_attempt(&self, generation: u64, why: &str) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let State::Armed(armed) = &mut *state else {
                return;
            };
            if armed.generation != generation {
                return;
            }
            armed.in_flight = false;
            armed.attempts_remaining = armed.attempts_remaining.saturating_sub(1);
            if armed.attempts_remaining == 0 {
                tracing::warn!("pairing: attempt budget exhausted — locked until re-armed ({why})");
                *state = State::Locked;
            } else {
                tracing::warn!(
                    remaining = armed.attempts_remaining,
                    "pairing: failed exchange attempt ({why})"
                );
            }
        }
        self.notify_changed();
    }

    /// Single-use success (spec §5.3): the first successful enrollment disarms.
    fn succeed(&self, generation: u64) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let State::Armed(armed) = &*state {
                if armed.generation == generation {
                    tracing::info!("pairing: enrollment succeeded — disarmed (single use)");
                    *state = State::Disarmed;
                }
            }
        }
        self.notify_changed();
    }
}

/// The pairing admin surface's join-fact bundle (spec §5.5 / §7): everything `PairingBegin`
/// needs beyond the manager to compose `PairingCode` — identity facts plus an injected
/// non-loopback address enumerator (the host crate stays free of interface-enumeration deps).
pub struct PairingSurface {
    /// The shared armed-state manager (the SAME instance wired into the `Authenticator`).
    pub manager: Arc<PairingManager>,
    /// The node's persistent id (lan-discovery spec §3.2).
    pub node_id: String,
    /// The node's display name (lan-discovery spec §3.3).
    pub node_name: String,
    /// The TLS listener's leaf-certificate SHA-256 (lowercase hex).
    pub server_fp: String,
    /// The TLS listener's bound port.
    pub port: u16,
    /// Enumerates the node's current non-loopback IP addresses, primary-preference first.
    pub addresses: Box<dyn Fn() -> Vec<std::net::IpAddr> + Send + Sync>,
}

impl PairingSurface {
    /// The `host:port` endpoint list (IPv6 bracketed), primary first.
    pub fn endpoints(&self) -> Vec<String> {
        (self.addresses)()
            .into_iter()
            .map(|ip| match ip {
                std::net::IpAddr::V4(v4) => format!("{v4}:{}", self.port),
                std::net::IpAddr::V6(v6) => format!("[{v6}]:{}", self.port),
            })
            .collect()
    }

    /// Compose the canonical `daemon+pair:` join URI (spec §7) for `code` over `endpoints`.
    /// The first endpoint is the URI authority; the rest ride as `alt` params.
    pub fn compose_uri(&self, code: &str, endpoints: &[String]) -> String {
        let primary = endpoints
            .first()
            .cloned()
            .unwrap_or_else(|| format!("127.0.0.1:{}", self.port));
        let mut uri = format!(
            "daemon+pair://{primary}/?v=1&code={code}&fp={}&node={}&name={}",
            self.server_fp,
            self.node_id,
            percent_encode(&self.node_name),
        );
        for alt in endpoints.iter().skip(1) {
            uri.push_str("&alt=");
            uri.push_str(alt);
        }
        uri
    }
}

/// Minimal percent-encoding for a URI query value: everything but RFC 3986 unreserved.
fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(*b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Arming/entropy failures surfaced to the admin caller (never to the SASL peer).
#[derive(Debug, thiserror::Error)]
pub enum PairingError {
    /// The OS CSPRNG failed — no code can be minted.
    #[error("no OS randomness for the pairing code: {0}")]
    Entropy(String),
}

/// An in-flight exchange between the node's `AuthChallenge` and the app's confirming `AuthStep`
/// (spec §4 message 3). Dropping it without completion counts a failed attempt — an abandoned
/// connection is a non-`AuthOk` outcome.
pub struct PendingPairing {
    manager: Arc<PairingManager>,
    finished: Finished,
    generation: u64,
    /// The handshake-observed client fingerprint (hex) — the identity that gets enrolled.
    client_fp: String,
    /// The sanitized device display name from the `AuthStart` payload.
    display_name: String,
}

impl PendingPairing {
    /// The owning manager (for the authenticator to signal the device-set change once the
    /// enrollment transaction lands — [`PairingManager::notify_changed`]).
    pub fn manager_handle(&self) -> Arc<PairingManager> {
        self.manager.clone()
    }

    /// The enrolled identity facts, exposed for the authenticator's store transaction.
    pub fn client_fingerprint(&self) -> &str {
        &self.client_fp
    }

    /// The sanitized device display name.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// Verify the app's confirmation MAC (`AuthStep` payload). `Ok` disarms the manager
    /// (single use) and hands back the enrollment facts; `Err` consumes one attempt.
    pub fn complete(self, step_payload: &[u8]) -> Result<(String, String), PairingRefusal> {
        let verified = daemon_api::from_cbor::<PairConfirm>(step_payload)
            .ok()
            .filter(|c| self.finished.verify_peer_confirmation(&c.ca))
            .is_some();
        // Deconstruct without running Drop (which would double-count the outcome).
        let this = std::mem::ManuallyDrop::new(self);
        let manager = this.manager.clone();
        let client_fp = this.client_fp.clone();
        let display_name = this.display_name.clone();
        if verified {
            manager.succeed(this.generation);
            Ok((client_fp, display_name))
        } else {
            manager.fail_attempt(this.generation, "confirmation MAC mismatch");
            Err(PairingRefusal::Failed)
        }
    }
}

impl Drop for PendingPairing {
    fn drop(&mut self) {
        self.manager
            .fail_attempt(self.generation, "exchange abandoned");
    }
}

/// The SPAKE2 responder step (spec §4 message 2): parse + validate the `AuthStart` payload,
/// produce the node share + confirmation MAC under the channel-binding AAD. Returns the CBOR
/// challenge, the finished exchange, and the sanitized device name.
fn run_spake2(
    w: &PasswordScalar,
    server_fp_hex: &str,
    client_fp_hex: &str,
    initial: &[u8],
) -> Result<(Vec<u8>, Finished, String), PairingRefusal> {
    let start = decode_start(initial).ok_or(PairingRefusal::Failed)?;
    let device_name = sanitize_device_name(&start.name);
    let aad =
        compose_aad(server_fp_hex, client_fp_hex, &device_name).ok_or(PairingRefusal::Failed)?;
    let exchange = Exchange::new_b(w, IDENT_APP, IDENT_NODE).map_err(|_| PairingRefusal::Failed)?;
    let pb = exchange.share().to_vec();
    let finished = exchange
        .finish(&start.pa, &aad)
        .map_err(|_| PairingRefusal::Failed)?;
    let challenge = daemon_api::to_cbor(&PairChallenge {
        pb,
        cb: finished.local_confirmation().to_vec(),
    });
    Ok((challenge, finished, device_name))
}

/// Lazily expire a stale armed code (TTL, spec §5.3).
fn expire_if_stale(state: &mut State) {
    if let State::Armed(armed) = state {
        if Instant::now() >= armed.deadline {
            tracing::info!("pairing: code expired — disarmed");
            *state = State::Disarmed;
        }
    }
}

/// 10 Crockford-base32 chars from the OS CSPRNG (≈50 bits; 256 % 32 == 0, so `% 32` is unbiased).
fn generate_code() -> Result<String, PairingError> {
    let mut bytes = [0u8; CODE_LEN];
    getrandom::getrandom(&mut bytes).map_err(|e| PairingError::Entropy(e.to_string()))?;
    Ok(bytes
        .iter()
        .map(|b| CROCKFORD[(b % 32) as usize] as char)
        .collect())
}

/// Display-group a canonical code as `XXXXX-XXXXX` (spec §5.3).
pub fn group_code(code: &str) -> String {
    if code.len() == CODE_LEN {
        format!("{}-{}", &code[..5], &code[5..])
    } else {
        code.to_string()
    }
}

/// Canonicalize a human-entered code: uppercase, separators stripped, Crockford confusables
/// folded (`I`/`L`→`1`, `O`→`0`) — the §3.2 normalization, shared with the app side.
pub fn canonicalize_code(input: &str) -> String {
    input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| match c.to_ascii_uppercase() {
            'I' | 'L' => '1',
            'O' => '0',
            up => up,
        })
        .collect()
}

/// Decode + validate the `AuthStart` payload: schema v1, a 65-byte uncompressed share, a
/// plausibly sized name (byte-bounded before sanitization).
fn decode_start(initial: &[u8]) -> Option<PairStart> {
    let start = daemon_api::from_cbor::<PairStart>(initial).ok()?;
    (start.v == 1 && start.pa.len() == SHARE_LEN && start.name.len() <= 256).then_some(start)
}

/// The channel-binding AAD (spec §3.3): raw server leaf fp ‖ raw client leaf fp ‖ device name.
fn compose_aad(server_fp_hex: &str, client_fp_hex: &str, device_name: &str) -> Option<Vec<u8>> {
    let mut aad = Vec::with_capacity(64 + device_name.len());
    aad.extend_from_slice(&hex_decode32(server_fp_hex)?);
    aad.extend_from_slice(&hex_decode32(client_fp_hex)?);
    aad.extend_from_slice(device_name.as_bytes());
    Some(aad)
}

/// Decode a 64-char lowercase-hex SHA-256 fingerprint into its raw 32 bytes.
fn hex_decode32(hex: &str) -> Option<[u8; 32]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out[i] = (hi as u8) << 4 | lo as u8;
    }
    Some(out)
}

/// Sanitize the device display name (spec §4): strip control characters, collapse surrounding
/// whitespace, cap at 64 characters; an empty result falls back to `"device"`.
fn sanitize_device_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .filter(|c| !c.is_control())
        .take(64)
        .collect::<String>()
        .trim()
        .to_string();
    if cleaned.is_empty() {
        "device".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVER_FP: &str = "aa11bb22cc33dd44ee55ff660011223344556677889900aabbccddeeff001122";
    const CLIENT_FP: &str = "1122334455667788990011223344556677889900112233445566778899001122";

    /// Compose the app-side (`AuthStart`) payload for `code`, returning the payload and the
    /// initiator exchange (to finish once the node's challenge arrives).
    fn app_start(code: &str, name: &str) -> (Vec<u8>, Exchange) {
        let w = PasswordScalar::derive(code.as_bytes());
        let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).expect("exchange a");
        #[derive(Serialize)]
        struct Start<'a> {
            v: u8,
            #[serde(with = "serde_bytes")]
            pa: &'a [u8],
            name: &'a str,
        }
        let payload = daemon_api::to_cbor(&Start {
            v: 1,
            pa: a.share(),
            name,
        });
        (payload, a)
    }

    #[derive(Deserialize)]
    struct Challenge {
        #[serde(with = "serde_bytes")]
        pb: Vec<u8>,
        #[serde(with = "serde_bytes")]
        cb: Vec<u8>,
    }

    fn aad(name: &str) -> Vec<u8> {
        compose_aad(SERVER_FP, CLIENT_FP, name).unwrap()
    }

    #[test]
    fn arm_status_cancel_lifecycle() {
        let m = Arc::new(PairingManager::new());
        assert!(!m.is_armed());
        let armed = m.arm().unwrap();
        assert_eq!(armed.code.len(), CODE_LEN);
        assert!(armed.code.bytes().all(|b| CROCKFORD.contains(&b)));
        let status = m.status();
        assert!(status.armed && !status.locked);
        assert_eq!(status.attempts_remaining, Some(PAIRING_ATTEMPTS));
        m.cancel();
        assert!(!m.is_armed());
        assert!(!m.status().locked);
    }

    #[test]
    fn full_exchange_enrolls_and_disarms() {
        let m = Arc::new(PairingManager::new());
        let armed = m.arm().unwrap();
        let (payload, app) = app_start(&armed.code, "Pixel 9");
        let (challenge, pending) = m
            .begin_exchange(SERVER_FP, CLIENT_FP, &payload)
            .expect("exchange starts");
        // App side: verify the node's MAC, produce ca.
        let ch: Challenge = daemon_api::from_cbor(&challenge).unwrap();
        let fin_a = app.finish(&ch.pb, &aad("Pixel 9")).expect("app finish");
        assert!(fin_a.verify_peer_confirmation(&ch.cb), "app verifies cb");
        #[derive(Serialize)]
        struct Confirm<'a> {
            #[serde(with = "serde_bytes")]
            ca: &'a [u8],
        }
        let step = daemon_api::to_cbor(&Confirm {
            ca: fin_a.local_confirmation(),
        });
        let (fp, name) = pending.complete(&step).expect("node verifies ca");
        assert_eq!(fp, CLIENT_FP);
        assert_eq!(name, "Pixel 9");
        // Single use: success disarmed.
        assert!(!m.is_armed());
        assert!(!m.status().locked);
    }

    #[test]
    fn wrong_code_yields_nonverifying_mac_and_counts_attempt() {
        let m = Arc::new(PairingManager::new());
        let _armed = m.arm().unwrap();
        let (payload, app) = app_start("AAAAAAAAAA", "Mallory");
        let (challenge, pending) = m
            .begin_exchange(SERVER_FP, CLIENT_FP, &payload)
            .expect("exchange starts even with a wrong code");
        let ch: Challenge = daemon_api::from_cbor(&challenge).unwrap();
        // The app (holding the wrong w) cannot verify the node's MAC…
        let fin_a = app.finish(&ch.pb, &aad("Mallory")).expect("app finish");
        assert!(
            !fin_a.verify_peer_confirmation(&ch.cb),
            "cb must not verify"
        );
        // …and its own ca does not verify node-side either.
        #[derive(Serialize)]
        struct Confirm<'a> {
            #[serde(with = "serde_bytes")]
            ca: &'a [u8],
        }
        let step = daemon_api::to_cbor(&Confirm {
            ca: fin_a.local_confirmation(),
        });
        assert!(pending.complete(&step).is_err());
        assert_eq!(m.status().attempts_remaining, Some(PAIRING_ATTEMPTS - 1));
    }

    #[test]
    fn attempt_budget_locks_and_rearm_clears() {
        let m = Arc::new(PairingManager::new());
        let _armed = m.arm().unwrap();
        for i in 0..PAIRING_ATTEMPTS {
            let before = m.status().attempts_remaining;
            // Malformed payloads count attempts.
            let refused = m.begin_exchange(SERVER_FP, CLIENT_FP, b"garbage");
            assert!(matches!(refused, Err(PairingRefusal::Failed)), "try {i}");
            let _ = before;
        }
        assert!(m.status().locked);
        // Locked refuses with the dedicated reason…
        let (payload, _) = app_start("AAAAAAAAAA", "x");
        assert!(matches!(
            m.begin_exchange(SERVER_FP, CLIENT_FP, &payload),
            Err(PairingRefusal::Locked)
        ));
        // …until an admin re-arms.
        let _ = m.arm().unwrap();
        assert!(m.is_armed());
        assert!(!m.status().locked);
    }

    #[test]
    fn concurrent_exchange_refused_without_consuming_attempt() {
        let m = Arc::new(PairingManager::new());
        let armed = m.arm().unwrap();
        let (payload, _app) = app_start(&armed.code, "first");
        let (_challenge, pending) = m.begin_exchange(SERVER_FP, CLIENT_FP, &payload).unwrap();
        let attempts_before = m.status().attempts_remaining;
        let (payload2, _app2) = app_start(&armed.code, "second");
        assert!(matches!(
            m.begin_exchange(SERVER_FP, CLIENT_FP, &payload2),
            Err(PairingRefusal::Failed)
        ));
        assert_eq!(m.status().attempts_remaining, attempts_before);
        // Abandoning the in-flight exchange (connection drop) counts one attempt and unblocks.
        drop(pending);
        assert_eq!(
            m.status().attempts_remaining,
            attempts_before.map(|a| a - 1)
        );
        let (payload3, _app3) = app_start(&armed.code, "third");
        assert!(m.begin_exchange(SERVER_FP, CLIENT_FP, &payload3).is_ok());
    }

    #[test]
    fn not_armed_and_rearm_invalidates_previous_code() {
        let m = Arc::new(PairingManager::new());
        let (payload, _) = app_start("AAAAAAAAAA", "x");
        assert!(matches!(
            m.begin_exchange(SERVER_FP, CLIENT_FP, &payload),
            Err(PairingRefusal::NotArmed)
        ));
        // Arm code 1, start an exchange, then re-arm: the stale exchange cannot mutate the new
        // armed state (its completion fails and the new code stays armed with a full budget).
        let armed1 = m.arm().unwrap();
        let (p1, app1) = app_start(&armed1.code, "stale");
        let (ch1, pending1) = m.begin_exchange(SERVER_FP, CLIENT_FP, &p1).unwrap();
        let _armed2 = m.arm().unwrap();
        let ch: Challenge = daemon_api::from_cbor(&ch1).unwrap();
        let fin = app1.finish(&ch.pb, &aad("stale")).unwrap();
        #[derive(Serialize)]
        struct Confirm<'a> {
            #[serde(with = "serde_bytes")]
            ca: &'a [u8],
        }
        let step = daemon_api::to_cbor(&Confirm {
            ca: fin.local_confirmation(),
        });
        // The stale exchange verifies cryptographically (same code semantics) but must not
        // disarm the NEW generation.
        let _ = pending1.complete(&step);
        assert!(m.is_armed(), "re-armed code survives a stale completion");
        assert_eq!(m.status().attempts_remaining, Some(PAIRING_ATTEMPTS));
    }

    #[test]
    fn code_canonicalization_folds_confusables() {
        assert_eq!(canonicalize_code("abcde-fghjk"), "ABCDEFGHJK");
        assert_eq!(canonicalize_code("AbC1I lO0o"), "ABC111000");
        assert_eq!(group_code("ABCDE01234"), "ABCDE-01234");
    }

    #[test]
    fn device_name_sanitized() {
        assert_eq!(sanitize_device_name("  Pixel\t 9\u{7}  "), "Pixel 9");
        assert_eq!(sanitize_device_name("\u{1}\u{2}"), "device");
        assert_eq!(sanitize_device_name(&"x".repeat(200)).len(), 64);
    }

    #[test]
    fn channel_binding_mitm_relay_fails() {
        // A MITM relays the exchange across two TLS legs: the app sees the MITM's server cert,
        // the node sees the MITM's client cert — the AADs differ, so neither MAC verifies.
        let m = Arc::new(PairingManager::new());
        let armed = m.arm().unwrap();
        let (payload, app) = app_start(&armed.code, "victim");
        let (challenge, pending) = m.begin_exchange(SERVER_FP, CLIENT_FP, &payload).unwrap();
        let ch: Challenge = daemon_api::from_cbor(&challenge).unwrap();
        // App-side AAD carries a DIFFERENT server fp (the MITM's cert on the app's leg).
        let mitm_fp = "9999999999999999999999999999999999999999999999999999999999999999";
        let app_aad = compose_aad(mitm_fp, CLIENT_FP, "victim").unwrap();
        let fin_a = app.finish(&ch.pb, &app_aad).unwrap();
        assert!(
            !fin_a.verify_peer_confirmation(&ch.cb),
            "app must not verify the node MAC across a relay"
        );
        #[derive(Serialize)]
        struct Confirm<'a> {
            #[serde(with = "serde_bytes")]
            ca: &'a [u8],
        }
        let step = daemon_api::to_cbor(&Confirm {
            ca: fin_a.local_confirmation(),
        });
        assert!(pending.complete(&step).is_err(), "node must not verify ca");
    }

    /// The §11 cross-implementation transcript fixture: the full §4 exchange with pinned
    /// ephemeral scalars, generated by THIS implementation and checked in at
    /// `tests/fixtures/pairing-transcript-v1.json` for the C++ side to replay — it catches
    /// encoding drift (CBOR payload shapes, AAD composition, transcript framing) that the RFC
    /// 9382 vectors alone would miss. Regenerate with `DAEMON_BLESS=1 cargo test -p daemon-host
    /// pairing::tests::transcript_fixture` after a DELIBERATE protocol change.
    #[test]
    fn transcript_fixture_matches_checked_in() {
        use daemon_pake::{scalar_from_bytes, Role};
        use sha2::{Digest, Sha256};

        fn hex(bytes: &[u8]) -> String {
            bytes.iter().map(|b| format!("{b:02x}")).collect()
        }
        // Pinned, self-documenting ephemerals: sha256 of a fixed label, reduced-range by luck of
        // the curve (P-256's order is close enough to 2^256 that these digests are in range —
        // asserted, not assumed).
        fn pinned_scalar(label: &str) -> daemon_pake::Scalar {
            let digest: [u8; 32] = Sha256::digest(label.as_bytes()).into();
            scalar_from_bytes(&digest).expect("pinned digest is < the group order")
        }

        let code = "ABCDE01234";
        let name = "Pixel 9";
        let w = PasswordScalar::derive(code.as_bytes());
        let x = pinned_scalar("daemon-pair fixture scalar x (app)");
        let y = pinned_scalar("daemon-pair fixture scalar y (node)");

        // Message 1 (app → node): the same CBOR shape production `decode_start` parses.
        let a = Exchange::with_scalar(Role::A, &w, x, IDENT_APP, IDENT_NODE);
        let pa = a.share().to_vec();
        #[derive(Serialize)]
        struct Start<'a> {
            v: u8,
            #[serde(with = "serde_bytes")]
            pa: &'a [u8],
            name: &'a str,
        }
        let auth_start = daemon_api::to_cbor(&Start {
            v: 1,
            pa: &pa,
            name,
        });

        // Message 2 (node → app): the production parse/AAD/encode path, with the pinned y where
        // `run_spake2` draws a fresh scalar.
        let start = decode_start(&auth_start).expect("fixture start parses");
        let device_name = sanitize_device_name(&start.name);
        let aad = compose_aad(SERVER_FP, CLIENT_FP, &device_name).unwrap();
        let b = Exchange::with_scalar(Role::B, &w, y, IDENT_APP, IDENT_NODE);
        let pb = b.share().to_vec();
        let fin_b = b.finish(&start.pa, &aad).expect("node finish");
        let auth_challenge = daemon_api::to_cbor(&PairChallenge {
            pb: pb.clone(),
            cb: fin_b.local_confirmation().to_vec(),
        });

        // Message 3 (app → node) + both sides' key agreement.
        let fin_a = a.finish(&pb, &aad).expect("app finish");
        assert!(fin_a.verify_peer_confirmation(fin_b.local_confirmation()));
        assert!(fin_b.verify_peer_confirmation(fin_a.local_confirmation()));
        assert_eq!(fin_a.key(), fin_b.key());
        #[derive(Serialize)]
        struct Confirm<'a> {
            #[serde(with = "serde_bytes")]
            ca: &'a [u8],
        }
        let auth_step = daemon_api::to_cbor(&Confirm {
            ca: fin_a.local_confirmation(),
        });

        let fixture = serde_json::json!({
            "suite": daemon_pake::SUITE,
            "code": code,
            "device_name": name,
            "server_fp": SERVER_FP,
            "client_fp": CLIENT_FP,
            "w": hex(&w.to_bytes()),
            "aad": hex(&aad),
            "auth_start_payload": hex(&auth_start),
            "pa": hex(&pa),
            "auth_challenge_payload": hex(&auth_challenge),
            "pb": hex(&pb),
            "cb": hex(fin_b.local_confirmation()),
            "auth_step_payload": hex(&auth_step),
            "ca": hex(fin_a.local_confirmation()),
            "ke": hex(fin_a.key()),
        });

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/pairing-transcript-v1.json");
        if std::env::var_os("DAEMON_BLESS").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, format!("{fixture:#}\n")).unwrap();
            return;
        }
        let checked_in: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect(
                "checked-in fixture missing — bless it: DAEMON_BLESS=1 cargo test -p \
                 daemon-host pairing::tests::transcript_fixture",
            ))
            .expect("fixture parses");
        assert_eq!(
            fixture, checked_in,
            "transcript drifted from the checked-in fixture — if the protocol change is \
             deliberate, re-bless AND update the C++ replay"
        );
    }
}
