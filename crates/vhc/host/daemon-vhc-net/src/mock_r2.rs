// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! An in-process mock of the coordinator presign endpoint + the R2 object store (NET-1/3/8),
//! built on `wiremock` (no live network). Mirrors what the cloud `apps/vhc` worker does:
//! `POST /api/v1/vhc/runs/:id/presign` returns a URL into a stateful `/obj/*` PUT/GET store at
//! the spec §11.3 object key. It can mint expired presigns (`with_expiry`), drop objects
//! (`evict`), and corrupt them in place (`corrupt`) for the negative cases.
//!
//! A REUSABLE HARNESS FIXTURE, never a production store: available to this crate's own suites
//! (`cfg(test)`) and — behind the `harness` feature — to downstream suites (the worker's
//! live-attach tests, the multi-process acceptance smoke), so every consumer exercises the same
//! presign/object contract. The dev R2 deployment remains the live-gated counterpart.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_egress::{EgressClient, EgressConfig};
use wiremock::matchers::{method, path_regex};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use crate::presign::{HttpPresignClient, PresignRequest, PresignResponse};
use crate::r2_store::r2_object_key;
use crate::seam::RunId;

type Objects = Arc<Mutex<HashMap<String, Vec<u8>>>>;

fn now_s() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The presign responder: parses the request body, computes the §11.3 object key, and returns a
/// URL into this server's `/obj/*` store with the configured expiry.
struct Presigner {
    base: String,
    expiry_offset_s: i64,
}

impl Respond for Presigner {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        let run = req
            .url
            .path()
            .trim_start_matches('/')
            .strip_prefix("api/v1/vhc/runs/")
            .and_then(|s| s.split('/').next())
            .unwrap_or_default()
            .to_string();
        let preq: PresignRequest = match serde_json::from_slice(&req.body) {
            Ok(p) => p,
            Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
        };
        let key = match r2_object_key(&RunId::new(run), &preq) {
            Ok(k) => k,
            Err(e) => return ResponseTemplate::new(400).set_body_string(e.to_string()),
        };
        let expires_at = (now_s() + self.expiry_offset_s).max(0) as u64;
        let resp = PresignResponse {
            url: format!("{}/obj/{key}?sig=mock", self.base),
            expires_at,
            headers: BTreeMap::new(),
        };
        ResponseTemplate::new(200).set_body_json(serde_json::to_value(&resp).expect("json"))
    }
}

struct PutObj {
    objects: Objects,
}
impl Respond for PutObj {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        self.objects
            .lock()
            .expect("objects mutex")
            .insert(req.url.path().to_string(), req.body.clone());
        ResponseTemplate::new(200)
    }
}

struct GetObj {
    objects: Objects,
}
impl Respond for GetObj {
    fn respond(&self, req: &Request) -> ResponseTemplate {
        match self
            .objects
            .lock()
            .expect("objects mutex")
            .get(req.url.path())
        {
            Some(bytes) => ResponseTemplate::new(200).set_body_bytes(bytes.clone()),
            None => ResponseTemplate::new(404),
        }
    }
}

/// A running mock coordinator + object store.
pub struct MockR2 {
    server: MockServer,
    objects: Objects,
}

impl MockR2 {
    /// Start with a healthy 15-minute presign expiry.
    pub async fn start() -> Self {
        Self::with_expiry(900).await
    }

    /// Start with presigns that expire `expiry_offset_s` seconds from now (negative = already
    /// expired — the expired-presign rejection case).
    pub async fn with_expiry(expiry_offset_s: i64) -> Self {
        let server = MockServer::start().await;
        let base = server.uri();
        let objects: Objects = Arc::new(Mutex::new(HashMap::new()));
        Mock::given(method("POST"))
            .and(path_regex(r"^/api/v1/vhc/runs/[^/]+/presign$"))
            .respond_with(Presigner {
                base: base.clone(),
                expiry_offset_s,
            })
            .mount(&server)
            .await;
        Mock::given(method("PUT"))
            .and(path_regex(r"^/obj/"))
            .respond_with(PutObj {
                objects: objects.clone(),
            })
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/obj/"))
            .respond_with(GetObj {
                objects: objects.clone(),
            })
            .mount(&server)
            .await;
        Self { server, objects }
    }

    /// The vhc coordinator base URL (`{uri}/api/v1/vhc`).
    pub fn coordinator_base(&self) -> String {
        format!("{}/api/v1/vhc", self.server.uri())
    }

    /// A fresh SSRF-safe egress client (the initial hop to the loopback mock is not re-checked).
    pub fn egress(&self) -> EgressClient {
        EgressClient::new(EgressConfig::default()).expect("egress client")
    }

    /// An [`HttpPresignClient`] pointed at this mock.
    pub fn presign_client(&self) -> HttpPresignClient {
        HttpPresignClient::new(self.egress(), self.coordinator_base())
    }

    /// Seed an object directly at its §11.3 key (bypassing PUT) — GET-only / artifact cases.
    pub fn seed(&self, object_key: &str, bytes: Vec<u8>) {
        self.objects
            .lock()
            .expect("objects mutex")
            .insert(format!("/obj/{object_key}"), bytes);
    }

    /// Drop a stored object (simulate lifecycle/retention expiry).
    pub fn evict(&self, object_key: &str) {
        self.objects
            .lock()
            .expect("objects mutex")
            .remove(&format!("/obj/{object_key}"));
    }

    /// Overwrite a stored object in place (simulate a tampering/broken store — the hash-verify
    /// negative cases).
    pub fn corrupt(&self, object_key: &str, bytes: &[u8]) {
        self.objects
            .lock()
            .expect("objects mutex")
            .insert(format!("/obj/{object_key}"), bytes.to_vec());
    }
}
