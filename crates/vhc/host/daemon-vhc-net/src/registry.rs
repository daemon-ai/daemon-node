// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`RegistryClient`] — run discovery + envelope fetch against the coordinator registry (spec
//! §6.1/§11.1; A1).
//!
//! The cloud `apps/vhc` worker exposes a validation-only run registry:
//! `GET {base}/runs` (snapshot) and `GET {base}/runs/:id` (one descriptor), each wrapped in
//! `{ "data": … }` (`apps/vhc/src/registry.ts`, `index.ts`). The descriptors carry the frozen
//! envelope's blake3 (`envelope_hash`) + artifact manifest, never the module bytes — the cloud
//! never fetches/executes a module (spec §11.1/§12), and every peer re-derives eligibility at
//! assess (§6.5).
//!
//! A node discovers a run here, then [`fetch_envelope`](RegistryClient::fetch_envelope)s the frozen
//! envelope object (presigned `GET` of `runs/<run>/envelope.cbor`, §11.3) and **blake3-verifies** it
//! against the descriptor's `envelope_hash` before handing the bytes to the worker's `AssessRun`.
//! All outbound HTTP rides the SSRF-safe [`EgressClient`] (raw `reqwest::Client` is clippy-banned);
//! auth is the same `vhc:*` credential the WS client uses (Bearer for the gateway, or the internal
//! identity headers for a direct-to-worker dev target) — never hardcoded.

use daemon_egress::{EgressClient, EgressRequest, Redirects};
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, to_canonical_vec, RosterDecision, RosterMutationResponse,
    RosterRecord, RosterSnapshot, SeatDecision, SeatLease, SeatMutationResponse, SeatRelease,
    SeatState,
};
use serde::{Deserialize, Serialize};

use crate::presign::{PresignOp, PresignRequest, PresignResponse};
use crate::seam::RunId;
use crate::VhcNetError;

/// One artifact the run references (name → pinned blake3 + size). Mirrors the cloud `ArtifactRef`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunArtifact {
    /// Run-relative artifact name/path (e.g. `experiment.wasm`).
    pub path: String,
    /// blake3 content hash, 64 lowercase hex chars.
    pub blake3: String,
    /// Declared size in bytes.
    pub size: u64,
}

/// A run descriptor from the registry (`apps/vhc` `RunDescriptor`). Experiment-opaque: it carries
/// the frozen envelope's hash + artifact manifest, never module bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunDescriptor {
    /// The run id (coordinator-assigned).
    pub run_id: String,
    /// Envelope schema major (spec §16).
    pub schema: u32,
    /// The vhc proto version the run is pinned to (§16).
    pub proto_version: u32,
    /// blake3 of the frozen envelope bytes (the signed hash), 64 lowercase hex chars.
    pub envelope_hash: String,
    /// ed25519 author public key (32 B), hex.
    pub author_pubkey: String,
    /// The envelope's artifact map (names + pinned hashes + sizes).
    #[serde(default)]
    pub artifacts: Vec<RunArtifact>,
    /// Per-peer round-payload cap in bytes.
    pub update_max_bytes: u64,
    /// Minimum roster size.
    pub min_peers: u32,
    /// Maximum roster size.
    pub max_peers: u32,
    /// Total rounds before the run finishes (`None` = driven elsewhere).
    #[serde(default)]
    pub rounds: Option<u64>,
    /// Creation time (unix seconds) stamped by the registry.
    #[serde(default)]
    pub created_at: u64,
    /// R2 key of the stored `envelope.cbor` (§11.3).
    #[serde(default)]
    pub envelope_key: String,
}

/// The `{ "data": T }` envelope every `apps/vhc` route wraps its success body in.
#[derive(Deserialize)]
struct DataEnvelope<T> {
    data: T,
}

/// One published-checkpoint pointer (spec §9), read from
/// `GET {base}/runs/:id/state`.`data.checkpoints[]`. Pointers are keyed **per `(role, kind)`**:
/// a role's restore source is scoped to that role's state (a coordinator pointer can never
/// shadow a trainer restore source), and within a role a periodic LIVE checkpoint is a distinct
/// slot from a graceful-leave DRAIN snapshot (restore prefers the freshest live pointer,
/// falling back to drain). Mirrors the cloud `CheckpointPointer`
/// (`apps/vhc/src/coordinator/checkpoint.ts`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPointer {
    /// The envelope role whose state the checkpoint captures.
    #[serde(default)]
    pub role: String,
    /// The pointer kind: [`CHECKPOINT_KIND_LIVE`] (periodic mid-run cadence) or
    /// [`CHECKPOINT_KIND_DRAIN`] (a graceful-leave drain snapshot).
    #[serde(default)]
    pub kind: String,
    /// The round the checkpoint captures (post-ingest state).
    pub round: u64,
    /// blake3 of the checkpoint bytes, 64 lowercase hex chars (the content address to verify).
    pub hash: String,
    /// The checkpoint byte length.
    #[serde(default)]
    pub size: u64,
    /// Whether ≥2 checkpointers uploaded byte-identical manifests (else registered-but-degraded).
    #[serde(default)]
    pub cross_checked: bool,
}

/// The periodic mid-run checkpoint kind (spec §9): published on the live cadence, so a
/// hard-crashed peer has a fresh restore source even though it never drained.
pub const CHECKPOINT_KIND_LIVE: &str = "live";

/// The graceful-leave drain-snapshot checkpoint kind (spec §9).
pub const CHECKPOINT_KIND_DRAIN: &str = "drain";

/// The coordinator run-state projection (`GET {base}/runs/:id/state`.`data`), the queryable surface
/// a rejoining peer reads for the current round + the latest checkpoint pointer (lane R). Only the
/// fields the rejoin path needs are decoded; the rest are ignored.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RunState {
    /// The coordinator phase (`waiting` / `warmup` / `round_train` / `cooldown` / …).
    #[serde(default)]
    pub phase: String,
    /// The coordinator's current round.
    #[serde(default)]
    pub round: u64,
    /// The current epoch.
    #[serde(default)]
    pub epoch: u64,
    /// Whether the run has finished.
    #[serde(default)]
    pub finished: bool,
    /// The published-checkpoint pointers, one per `(role, kind)` slot (empty = none published
    /// yet; a rejoining peer falls back to fresh-state, §9 first-epoch).
    #[serde(default)]
    pub checkpoints: Vec<CheckpointPointer>,
}

/// The outcome of a seat claim/renew against the registry's fencing-token compare-and-swap.
///
/// `Won` is a **storage** outcome, not an authority grant (the registry is untrusted): the
/// claimant may start/continue coordinating, but every peer independently verifies the stored
/// lease's signature, certificate chain, and supersession floor. `Lost` carries the registry's
/// structural refusal plus the slot's current state — the re-read a losing claimant needs to
/// decide between standing by and retrying at the floor + 1 after expiry.
#[derive(Clone, Debug, PartialEq, Eq)]
// A transient return value consumed at the call site, never stored in bulk — the size skew
// between the variants is not worth indirection on the client surface.
#[allow(clippy::large_enum_variant)]
pub enum SeatClaimOutcome {
    /// The CAS accepted; the slot now stores this lease.
    Won(SeatLease),
    /// The CAS refused; the slot is unchanged.
    Lost {
        /// The registry's structural refusal.
        decision: SeatDecision,
        /// The slot's current state (the incumbent lease, or unclaimed + tombstone floor).
        state: SeatState,
    },
}

/// The outcome of a roster publish against the registry's monotonic freshness upsert.
///
/// `Accepted` is a **storage** outcome, not an authority grant (the registry is untrusted):
/// peers verify every fetched record themselves. `Refused` carries the registry's structural
/// verdict plus the slot's stored record — the re-read a stale publisher needs to republish at a
/// fresher `(incarnation, issued_at_ms)` key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RosterPublishOutcome {
    /// The upsert accepted; the slot now stores this record.
    Accepted,
    /// The upsert refused; the slot is unchanged.
    Refused {
        /// The registry's structural refusal.
        decision: RosterDecision,
        /// The slot's stored record (boxed for variant-size hygiene; `None` on a structural
        /// refusal against an empty slot).
        stored: Option<Box<RosterRecord>>,
    },
}

/// The `vhc:*` credential the registry + presign requests carry (never hardcoded — sourced from
/// `JoinRun.credentials` / node config, mirroring [`crate::ws_client::WsAuth`]).
#[derive(Clone, Debug, Default)]
enum Auth {
    #[default]
    None,
    Bearer(String),
    Internal {
        org_id: String,
        actor: String,
    },
}

/// Discovery + envelope-fetch client against a coordinator registry base
/// (e.g. `https://api.daemon.ai/api/v1/vhc`).
pub struct RegistryClient {
    egress: EgressClient,
    base_url: String,
    auth: Auth,
}

impl RegistryClient {
    /// Build a client against `base_url` (a trailing `/` is trimmed).
    pub fn new(egress: EgressClient, base_url: impl Into<String>) -> Self {
        Self {
            egress,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            auth: Auth::None,
        }
    }

    /// Attach the `vhc:*`-scoped API-key bearer token (the gateway path).
    #[must_use]
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.auth = Auth::Bearer(token.into());
        self
    }

    /// Attach the internal identity headers (the direct-to-`apps/vhc` dev path).
    #[must_use]
    pub fn with_internal(mut self, org_id: impl Into<String>, actor: impl Into<String>) -> Self {
        self.auth = Auth::Internal {
            org_id: org_id.into(),
            actor: actor.into(),
        };
        self
    }

    /// The registry base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Discover all runs (`GET {base}/runs`).
    pub async fn list_runs(&self) -> Result<Vec<RunDescriptor>, VhcNetError> {
        let url = format!("{}/runs", self.base_url);
        let body = self.authed_get(&url).await?;
        let env: DataEnvelope<Vec<RunDescriptor>> = serde_json::from_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode run list: {e}")))?;
        Ok(env.data)
    }

    /// Fetch one run descriptor (`GET {base}/runs/:id`); `Ok(None)` on a 404.
    pub async fn get_run(&self, run_id: &str) -> Result<Option<RunDescriptor>, VhcNetError> {
        let url = format!("{}/runs/{run_id}", self.base_url);
        let req = self.authed_request(EgressRequest::get(&url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("get run {run_id}: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(None);
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read run {run_id}: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "get run {run_id} returned {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        let env: DataEnvelope<RunDescriptor> = serde_json::from_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode run {run_id}: {e}")))?;
        Ok(Some(env.data))
    }

    /// Fetch the coordinator run-state projection (`GET {base}/runs/:id/state`), the queryable
    /// surface a rejoining peer reads for the current round + the latest checkpoint pointer (spec
    /// §9; lane R). `Ok(None)` on a 404 (run not initialized).
    pub async fn fetch_state(&self, run_id: &str) -> Result<Option<RunState>, VhcNetError> {
        let url = format!("{}/runs/{run_id}/state", self.base_url);
        let req = self.authed_request(EgressRequest::get(&url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("get state {run_id}: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(None);
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read state {run_id}: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "get state {run_id} returned {status}"
            )));
        }
        let env: DataEnvelope<RunState> = serde_json::from_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode state {run_id}: {e}")))?;
        Ok(Some(env.data))
    }

    /// Publish this peer's latest checkpoint manifest to the coordinator (spec §9; lane R):
    /// `POST {base}/runs/:id/checkpoint` with `{role, kind, round, hash, size}` — the pointer is
    /// tracked per `(role, kind)` slot. The coordinator keeps the latest pointer per slot +
    /// cross-checks a two-checkpointer both-match (RUN-6). Best-effort — a failure is a soft
    /// warning (the pointer is advisory; the run is unaffected).
    pub async fn publish_checkpoint(
        &self,
        run_id: &str,
        role: &str,
        kind: &str,
        round: u64,
        hash: &str,
        size: u64,
    ) -> Result<(), VhcNetError> {
        let url = format!("{}/runs/{run_id}/checkpoint", self.base_url);
        let body = serde_json::json!({
            "role": role, "kind": kind, "round": round, "hash": hash, "size": size
        });
        let ereq = EgressRequest::post_json(&url, &body)
            .map_err(|e| VhcNetError::Transport(format!("encode checkpoint pointer: {e}")))?;
        let resp = self
            .egress
            .execute(self.authed_request(ereq), Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("publish checkpoint {run_id}: {e}")))?;
        if !resp.status().is_success() {
            return Err(VhcNetError::Transport(format!(
                "publish checkpoint {run_id} returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Fetch the frozen envelope for `run` and **blake3-verify** it against `descriptor.envelope_hash`.
    ///
    /// Presigns a `GET` of the run-relative `envelope.cbor` artifact (§11.3), downloads the bytes via
    /// [`EgressClient`], and rejects a hash mismatch as [`VhcNetError::HashMismatch`] (the tamper
    /// path, §12) — so a registry that served the wrong envelope can never reach `AssessRun`.
    pub async fn fetch_envelope(
        &self,
        run: &RunId,
        descriptor: &RunDescriptor,
    ) -> Result<Vec<u8>, VhcNetError> {
        let presigned = self
            .presign(
                run,
                &PresignRequest::artifact(PresignOp::Get, "envelope.cbor"),
            )
            .await?;
        // The presigned URL carries its own credential (SigV4 query / object-proxy HMAC), so the
        // object GET needs no auth headers — just the bytes.
        let resp = self
            .egress
            .get(&presigned.url, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Fetch(format!("fetch envelope: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Fetch(format!("read envelope body: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Fetch(format!(
                "envelope fetch returned {status}"
            )));
        }
        let got = blake3_hash(&body[..]).to_hex();
        if got != descriptor.envelope_hash {
            return Err(VhcNetError::HashMismatch {
                expected: descriptor.envelope_hash.clone(),
                actual: got,
            });
        }
        Ok(body.to_vec())
    }

    /// Read a run's seat slot for `role` (`GET {base}/runs/:id/seat/:role`): the stored
    /// [`SeatState`] — a signed lease, or unclaimed with the fencing-token tombstone floor a
    /// prospective claimant bids `floor + 1` against. `Ok(None)` on a 404 (unknown run).
    ///
    /// The registry stores and CASes the signed object but creates no authority: the caller MUST
    /// verify a returned lease itself (`SeatLease::authorize` against the genesis-trusted bases,
    /// plus the revocation/supersession judgment) before dialing its endpoint or accepting its
    /// claimant's records.
    pub async fn read_seat(
        &self,
        run: &RunId,
        role: &str,
    ) -> Result<Option<SeatState>, VhcNetError> {
        let url = format!("{}/runs/{}/seat/{role}", self.base_url, run.as_str());
        let req = self.authed_request(EgressRequest::get(&url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("read seat {role}: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(None);
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read seat body {role}: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "read seat {role} returned {status}"
            )));
        }
        let state: SeatState = from_canonical_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode seat state {role}: {e}")))?;
        Ok(Some(state))
    }

    /// Claim (or take over) a run's seat (`PUT {base}/runs/:id/seat/:role` with the canonical-CBOR
    /// signed lease). The registry CASes on the fencing token — classic compare-and-set with
    /// increment: an unclaimed/expired slot accepts exactly `floor + 1`. A lost CAS returns
    /// [`SeatClaimOutcome::Lost`] carrying the slot's current state (re-read included); the loser
    /// either accepts the incumbent or, once it expires, retries at the floor + 1.
    pub async fn claim_seat(
        &self,
        run: &RunId,
        lease: &SeatLease,
    ) -> Result<SeatClaimOutcome, VhcNetError> {
        let url = format!(
            "{}/runs/{}/seat/{}",
            self.base_url,
            run.as_str(),
            lease.body.role
        );
        let bytes = to_canonical_vec(lease)
            .map_err(|e| VhcNetError::Transport(format!("encode seat lease: {e}")))?;
        let req = EgressRequest::put(&url, bytes).header("content-type", "application/cbor");
        self.seat_mutation(req, lease, "claim seat").await
    }

    /// Renew (heartbeat) a held lease (`POST {base}/runs/:id/seat/:role/heartbeat` with the
    /// re-signed canonical-CBOR lease): same claimant, same incarnation, same fencing token; the
    /// fresh body extends `expires_at_ms` and may rebind the epoch (the epoch-rebind rule — a
    /// renew, never a takeover). A refusal means the seat moved: the claimant is fenced and must
    /// stop acting as coordinator.
    pub async fn renew_seat(
        &self,
        run: &RunId,
        lease: &SeatLease,
    ) -> Result<SeatClaimOutcome, VhcNetError> {
        let url = format!(
            "{}/runs/{}/seat/{}/heartbeat",
            self.base_url,
            run.as_str(),
            lease.body.role
        );
        let bytes = to_canonical_vec(lease)
            .map_err(|e| VhcNetError::Transport(format!("encode seat renew: {e}")))?;
        let req = EgressRequest::post(&url, bytes).header("content-type", "application/cbor");
        self.seat_mutation(req, lease, "renew seat").await
    }

    /// Release a held seat (`DELETE {base}/runs/:id/seat/:role` with the canonical-CBOR signed
    /// release). The slot transitions to unclaimed but retains the fencing-token floor (tokens
    /// never reset). `Ok` only on an accepted release; a refusal (the seat already moved — a
    /// benign race with a takeover) surfaces as a typed transport error the caller may ignore.
    pub async fn release_seat(
        &self,
        run: &RunId,
        role: &str,
        release: &SeatRelease,
    ) -> Result<(), VhcNetError> {
        let url = format!("{}/runs/{}/seat/{role}", self.base_url, run.as_str());
        let bytes = to_canonical_vec(release)
            .map_err(|e| VhcNetError::Transport(format!("encode seat release: {e}")))?;
        let req = self.authed_request(
            EgressRequest::delete(&url, bytes).header("content-type", "application/cbor"),
        );
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("release seat {role}: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read release body {role}: {e}")))?;
        if status.is_success() {
            return Ok(());
        }
        let rendered = from_canonical_slice::<SeatMutationResponse>(&body)
            .map(|r| format!("{:?}", r.decision))
            .unwrap_or_else(|_| String::from_utf8_lossy(&body).into_owned());
        Err(VhcNetError::Transport(format!(
            "release seat {role} returned {status}: {rendered}"
        )))
    }

    /// Publish this node's iroh roster record for a run (`PUT {base}/runs/:id/roster` with the
    /// canonical-CBOR signed record). The registry applies the normative monotonic freshness
    /// upsert (structural only — it stores, never judges authority). A 409 refusal comes back
    /// typed with the slot's stored record (the re-read a stale publisher needs to bid fresher).
    pub async fn publish_roster(
        &self,
        run: &RunId,
        record: &RosterRecord,
    ) -> Result<RosterPublishOutcome, VhcNetError> {
        let url = format!("{}/runs/{}/roster", self.base_url, run.as_str());
        let bytes = to_canonical_vec(record)
            .map_err(|e| VhcNetError::Transport(format!("encode roster record: {e}")))?;
        let req = self.authed_request(
            EgressRequest::put(&url, bytes).header("content-type", "application/cbor"),
        );
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("publish roster: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read roster publish body: {e}")))?;
        if status.is_success() {
            return Ok(RosterPublishOutcome::Accepted);
        }
        if status.as_u16() == 409 {
            let refusal: RosterMutationResponse = from_canonical_slice(&body)
                .map_err(|e| VhcNetError::Transport(format!("decode roster refusal (409): {e}")))?;
            return Ok(RosterPublishOutcome::Refused {
                decision: refusal.decision,
                stored: refusal.record.map(Box::new),
            });
        }
        Err(VhcNetError::Transport(format!(
            "publish roster returned {status}: {}",
            String::from_utf8_lossy(&body)
        )))
    }

    /// Fetch a run's roster snapshot (`GET {base}/runs/:id/roster`): every stored record, as the
    /// registry holds them. The caller MUST verify each entry itself (`RosterRecord::authorize`
    /// against the genesis-trusted bases + the freshness precedence) before trusting an address —
    /// the registry stores, it never vouches. `Ok(empty)` on a 404 (unknown run / no roster yet).
    pub async fn fetch_roster(&self, run: &RunId) -> Result<Vec<RosterRecord>, VhcNetError> {
        let url = format!("{}/runs/{}/roster", self.base_url, run.as_str());
        let req = self.authed_request(EgressRequest::get(&url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("fetch roster: {e}")))?;
        let status = resp.status();
        if status.as_u16() == 404 {
            return Ok(Vec::new());
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read roster body: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "fetch roster returned {status}"
            )));
        }
        let snapshot: RosterSnapshot = from_canonical_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode roster snapshot: {e}")))?;
        Ok(snapshot.entries)
    }

    /// Issue one seat mutation (claim/renew) and map the frozen status contract: 2xx + `Accepted`
    /// ⇒ [`SeatClaimOutcome::Won`]; 409 + a decoded [`SeatMutationResponse`] ⇒
    /// [`SeatClaimOutcome::Lost`] with the refusal and the slot's current state; anything else is
    /// a transport error.
    async fn seat_mutation(
        &self,
        req: EgressRequest,
        lease: &SeatLease,
        what: &str,
    ) -> Result<SeatClaimOutcome, VhcNetError> {
        let resp = self
            .egress
            .execute(self.authed_request(req), Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("{what}: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read {what} body: {e}")))?;
        if status.is_success() {
            return Ok(SeatClaimOutcome::Won(lease.clone()));
        }
        if status.as_u16() == 409 {
            let refusal: SeatMutationResponse = from_canonical_slice(&body)
                .map_err(|e| VhcNetError::Transport(format!("decode {what} refusal (409): {e}")))?;
            return Ok(SeatClaimOutcome::Lost {
                decision: refusal.decision,
                state: refusal.state,
            });
        }
        Err(VhcNetError::Transport(format!(
            "{what} returned {status}: {}",
            String::from_utf8_lossy(&body)
        )))
    }

    /// Presign one object for `run` (`POST {base}/runs/:id/presign`) with the registry auth applied.
    async fn presign(
        &self,
        run: &RunId,
        req: &PresignRequest,
    ) -> Result<PresignResponse, VhcNetError> {
        let url = format!("{}/runs/{}/presign", self.base_url, run.as_str());
        let ereq = EgressRequest::post_json(&url, req)
            .map_err(|e| VhcNetError::Transport(format!("encode presign request: {e}")))?;
        let resp = self
            .egress
            .execute(self.authed_request(ereq), Redirects::DEFAULT)
            .await
            .map_err(|e| VhcNetError::Transport(format!("presign request: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read presign body: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "presign endpoint returned {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        serde_json::from_slice(&body)
            .map_err(|e| VhcNetError::Transport(format!("decode presign response: {e}")))
    }

    /// Issue an authed GET and return the body bytes (2xx only).
    async fn authed_get(&self, url: &str) -> Result<Vec<u8>, VhcNetError> {
        let req = self.authed_request(EgressRequest::get(url));
        let resp = self
            .egress
            .execute(req, Redirects::None)
            .await
            .map_err(|e| VhcNetError::Transport(format!("registry GET {url}: {e}")))?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| VhcNetError::Transport(format!("read {url}: {e}")))?;
        if !status.is_success() {
            return Err(VhcNetError::Transport(format!(
                "registry GET {url} returned {status}"
            )));
        }
        Ok(body.to_vec())
    }

    /// Apply the configured auth headers to an outbound request.
    fn authed_request(&self, req: EgressRequest) -> EgressRequest {
        match &self.auth {
            Auth::None => req,
            Auth::Bearer(token) => req.bearer_auth(token),
            Auth::Internal { org_id, actor } => req
                .header("x-daemon-org-id", org_id)
                .header("x-daemon-actor", actor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_descriptor_decodes_registry_shape() {
        // The `{ "data": … }`-wrapped shape `apps/vhc` returns, with optional fields present.
        let json = r#"{
            "run_id": "run-1", "schema": 1, "proto_version": 3,
            "envelope_hash": "aa", "author_pubkey": "bb",
            "artifacts": [{"path": "envelope.cbor", "blake3": "cc", "size": 12}],
            "update_max_bytes": 1048576, "min_peers": 1, "max_peers": 8,
            "rounds": 10, "created_at": 42, "envelope_key": "runs/run-1/envelope.cbor"
        }"#;
        let d: RunDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.run_id, "run-1");
        assert_eq!(d.proto_version, 3);
        assert_eq!(d.rounds, Some(10));
        assert_eq!(d.artifacts.len(), 1);
    }

    #[test]
    fn run_state_decodes_checkpoint_pointers_and_empty() {
        // The `GET /state` shape (spec §9): the per-(role, kind) `data.checkpoints` pointers a
        // rejoining peer reads.
        let with = r#"{"data":{"phase":"round_train","round":6,"epoch":1,"finished":false,
            "roster":["aa"],"committed":[],"coord_pubkey":"bb",
            "checkpoints":[
              {"role":"trainer","kind":"live","round":5,"hash":"dd","size":2048,"cross_checked":false,"uploads":1},
              {"role":"trainer","kind":"drain","round":3,"hash":"cc","size":4096,"cross_checked":true,"uploads":2}
            ]}}"#;
        let env: DataEnvelope<RunState> = serde_json::from_str(with).unwrap();
        let s = env.data;
        assert_eq!(s.phase, "round_train");
        assert_eq!(s.round, 6);
        assert_eq!(s.checkpoints.len(), 2);
        assert_eq!(s.checkpoints[0].role, "trainer");
        assert_eq!(s.checkpoints[0].kind, CHECKPOINT_KIND_LIVE);
        assert_eq!(s.checkpoints[0].round, 5);
        assert_eq!(s.checkpoints[1].kind, CHECKPOINT_KIND_DRAIN);
        assert!(s.checkpoints[1].cross_checked);

        // No checkpoint published yet → empty (the fresh-state fallback trigger).
        let without = r#"{"data":{"phase":"waiting","round":0,"epoch":0,"finished":false,
            "roster":[],"committed":[],"coord_pubkey":"bb","checkpoints":[]}}"#;
        let env2: DataEnvelope<RunState> = serde_json::from_str(without).unwrap();
        assert!(env2.data.checkpoints.is_empty());
    }
}
