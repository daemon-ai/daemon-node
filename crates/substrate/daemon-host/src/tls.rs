// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The TLS TCP transport for the node api surface (deliverable (2) of the Auth 3 track).
//!
//! The Unix-domain socket ([`crate::socket`]) stays plaintext (local trust); any *networked* api
//! access goes over TLS and **always requires authentication**. [`serve_api_tls_tcp`] accepts TCP
//! connections, performs the `rustls` handshake (server cert always; client cert verified when
//! `require_client_cert` is set, i.e. mTLS), captures the peer-certificate fingerprint for the
//! EXTERNAL mechanism, and then runs an authenticated multiplexed loop: a connection must complete a
//! SASL exchange (driven by the [`Authenticator`]) before any `Call`/`Open` is served — a pre-auth
//! request resolves to [`ApiError::Unauthenticated`] and the connection stays unelevated.
//!
//! Crypto provider: this module pins the **aws-lc-rs** `rustls` provider, matching the provider the
//! rest of the dependency tree already resolves (`cargo tree -i rustls` -> rustls 0.23 + aws-lc-rs),
//! so no second crypto backend is introduced.
//!
//! Scope boundary (held for the convergence step, deliverable (3)): the per-request **authorization
//! gate + request context** are *not* applied here yet — they depend on Track A (Auth 2), which is
//! not merged. The dispatch site below carries a `TODO(auth3-deliverable3)` marking exactly where
//! `with_request_context(principal)` + `authorize(&req)` will wrap `dispatch`. This module also does
//! **not** modify the existing Unix [`crate::socket::serve_mux`]; unifying the two loops under the
//! authenticated state machine is likewise part of the convergence step.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use daemon_api::{
    dispatch, from_cbor, is_streaming, to_cbor, ApiError, ApiResponse, NodeApi, WireC2S, WireS2C,
    WIRE_FEATURE_AUTH, WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM, WIRE_FEATURE_VERSIONING,
    WIRE_VERSION,
};
use daemon_auth::Principal;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio_rustls::rustls::pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio_rustls::rustls::server::WebPkiClientVerifier;
use tokio_rustls::rustls::{RootCertStore, ServerConfig};
use tokio_rustls::TlsAcceptor;

use crate::authn::{AuthExchange, AuthSuccess, Authenticator, BeginOutcome, StepOutcome, TlsState};
use crate::socket::{read_frame, spawn_stream, write_frame, WRITER_QUEUE};

/// Resolved `[api]` TLS configuration (the cert/key + client-auth policy). Built by `bins/daemon`
/// from the `[api]` table and handed to [`build_server_config`].
#[derive(Clone, Debug)]
pub struct ApiTlsConfig {
    /// PEM file with the server certificate chain.
    pub cert_path: PathBuf,
    /// PEM file with the server private key (PKCS#8 / SEC1 / PKCS#1).
    pub key_path: PathBuf,
    /// Require + verify a client certificate (mTLS). Enables EXTERNAL; rejects untrusted client
    /// certs at the TLS layer.
    pub require_client_cert: bool,
    /// PEM bundle of CA certificates trusted to sign client certificates. Required when
    /// `require_client_cert` is set.
    pub client_ca_path: Option<PathBuf>,
}

/// What can go wrong building the TLS [`ServerConfig`].
#[derive(Debug, thiserror::Error)]
pub enum TlsConfigError {
    /// A PEM cert/key file could not be read or parsed.
    #[error("reading {path}: {source}")]
    Pem {
        /// The file that could not be read/parsed.
        path: String,
        /// The underlying PEM error.
        source: tokio_rustls::rustls::pki_types::pem::Error,
    },
    /// `require_client_cert` was set without a `tls_client_ca` bundle to verify against.
    #[error("require_client_cert is set but no tls_client_ca was configured")]
    MissingClientCa,
    /// A rustls configuration error (bad cert/key, etc.).
    #[error("tls: {0}")]
    Rustls(#[from] tokio_rustls::rustls::Error),
    /// The client-certificate verifier could not be built (e.g. an unparsable CA bundle).
    #[error("client cert verifier: {0}")]
    Verifier(String),
}

/// Load a PEM certificate chain via the `rustls-pki-types` `PemObject` reader (the maintained
/// replacement for the archived `rustls-pemfile`).
fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, TlsConfigError> {
    let pem = |source| TlsConfigError::Pem {
        path: path.display().to_string(),
        source,
    };
    CertificateDer::pem_file_iter(path)
        .map_err(pem)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(pem)
}

/// Load a PEM private key (PKCS#8 / SEC1 / PKCS#1) via `PemObject`.
fn load_key(path: &Path) -> Result<PrivateKeyDer<'static>, TlsConfigError> {
    PrivateKeyDer::from_pem_file(path).map_err(|source| TlsConfigError::Pem {
        path: path.display().to_string(),
        source,
    })
}

/// Build a rustls [`ServerConfig`] from the resolved [`ApiTlsConfig`], pinning the aws-lc-rs crypto
/// provider (matching the rest of the tree). With `require_client_cert`, an mTLS verifier is built
/// over the configured client-CA bundle so untrusted client certificates are rejected during the
/// handshake; otherwise client certs are optional (clients authenticate via SCRAM/PLAIN over the
/// server-authenticated channel, and a presented cert is still captured for EXTERNAL).
pub fn build_server_config(cfg: &ApiTlsConfig) -> Result<Arc<ServerConfig>, TlsConfigError> {
    let provider = Arc::new(tokio_rustls::rustls::crypto::aws_lc_rs::default_provider());
    let certs = load_certs(&cfg.cert_path)?;
    let key = load_key(&cfg.key_path)?;

    let builder = ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()?;

    let config = if cfg.require_client_cert {
        let ca_path = cfg
            .client_ca_path
            .as_ref()
            .ok_or(TlsConfigError::MissingClientCa)?;
        let mut roots = RootCertStore::empty();
        for ca in load_certs(ca_path)? {
            roots.add(ca)?;
        }
        let verifier = WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider)
            .build()
            .map_err(|e| TlsConfigError::Verifier(e.to_string()))?;
        builder
            .with_client_cert_verifier(verifier)
            .with_single_cert(certs, key)?
    } else {
        builder.with_no_client_auth().with_single_cert(certs, key)?
    };
    Ok(Arc::new(config))
}

/// Lower-hex encode bytes (certificate fingerprints).
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The SHA-256 fingerprint (hex) of the peer's leaf certificate, if one was presented + verified.
fn peer_fingerprint<IO>(stream: &tokio_rustls::server::TlsStream<IO>) -> Option<String> {
    let (_, conn) = stream.get_ref();
    conn.peer_certificates()
        .and_then(|certs| certs.first())
        .map(|cert| hex(&Sha256::digest(cert.as_ref())))
}

/// Serve the node api surface over TLS/TCP until the listener errors. Every connection is mux-only
/// and **must authenticate** (TCP is never local-trusted). Spawn it as a background task alongside
/// the Unix listener.
pub async fn serve_api_tls_tcp(
    listener: TcpListener,
    tls: Arc<ServerConfig>,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
) {
    let acceptor = TlsAcceptor::from(tls);
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let acceptor = acceptor.clone();
                let api = api.clone();
                let auth = auth.clone();
                tokio::spawn(async move {
                    match acceptor.accept(stream).await {
                        Ok(tls_stream) => {
                            let tls_state = TlsState {
                                is_tls: true,
                                peer_cert_fingerprint: peer_fingerprint(&tls_stream),
                            };
                            let (rd, wr) = tokio::io::split(tls_stream);
                            if let Err(e) =
                                serve_authenticated_mux(rd, wr, api, auth, tls_state).await
                            {
                                tracing::debug!("tls api connection ended: {e}");
                            }
                        }
                        // A failed handshake (untrusted client cert under mTLS, protocol mismatch,
                        // a plaintext probe) is dropped cleanly — never panics the accept loop.
                        Err(e) => tracing::debug!("tls handshake failed: {e}"),
                    }
                });
            }
            Err(e) => {
                tracing::warn!("tls api accept failed: {e}");
                return;
            }
        }
    }
}

/// The per-connection authentication state.
enum ConnAuth {
    /// No successful auth yet; `Call`/`Open` are refused with `Unauthenticated`.
    Unauthenticated,
    /// A multi-step mechanism (SCRAM) is mid-exchange.
    InProgress(AuthExchange),
    /// Authenticated; the bound principal gates dispatch now and, in deliverable (3), will be
    /// threaded into the Track-A request context + authorize gate at the dispatch site.
    // The principal is intentionally retained but not yet read (see the `TODO(auth3-deliverable3)`
    // at the `Call` dispatch site); the convergence step consumes it.
    #[allow(dead_code)]
    Authenticated(Principal),
}

/// Deliver a completed authentication to the client: any trailing mechanism bytes (the SCRAM
/// server-final message) ride a final `AuthChallenge` before `AuthOk` (the frozen `AuthOk` carries
/// no mechanism bytes), then `AuthOk { token, principal }`. Returns the bound principal.
async fn complete_auth(
    tx: &mpsc::Sender<WireS2C>,
    final_data: Option<Vec<u8>>,
    success: Box<AuthSuccess>,
) -> Principal {
    if let Some(data) = final_data {
        let _ = tx.send(WireS2C::AuthChallenge { data }).await;
    }
    let AuthSuccess {
        principal,
        token,
        principal_view,
    } = *success;
    let _ = tx
        .send(WireS2C::AuthOk {
            token,
            principal: principal_view,
        })
        .await;
    principal
}

/// The authenticated mux loop, generic over the (TLS) byte stream halves. Mirrors
/// [`crate::socket::serve_mux`] but gates every `Call`/`Open` behind a completed SASL exchange.
async fn serve_authenticated_mux<R, W>(
    mut rd: R,
    wr: W,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
    tls_state: TlsState,
) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (tx, mut rx) = mpsc::channel::<WireS2C>(WRITER_QUEUE);
    let writer = tokio::spawn(async move {
        let mut wr = wr;
        while let Some(frame) = rx.recv().await {
            if write_frame(&mut wr, &to_cbor(&frame)).await.is_err() {
                break;
            }
        }
    });

    let mut streams: HashMap<u64, AbortHandle> = HashMap::new();
    let mut conn = ConnAuth::Unauthenticated;

    loop {
        let bytes = match read_frame(&mut rd).await? {
            Some(b) => b,
            None => break,
        };
        let frame = match from_cbor::<WireC2S>(&bytes) {
            Ok(f) => f,
            // Undecodable frames are dropped rather than killing the connection.
            Err(_) => continue,
        };
        match frame {
            WireC2S::Hello { .. } => {
                let mut features = vec![
                    WIRE_FEATURE_MUX.to_string(),
                    WIRE_FEATURE_STREAM.to_string(),
                ];
                if api.supports_versioning() {
                    features.push(WIRE_FEATURE_VERSIONING.to_string());
                }
                // TCP always requires auth: advertise the auth feature + the permitted mechanisms.
                features.push(WIRE_FEATURE_AUTH.to_string());
                let _ = tx
                    .send(WireS2C::Hello {
                        wire_version: WIRE_VERSION,
                        features,
                        auth_mechanisms: auth.advertised_mechanisms(&tls_state),
                    })
                    .await;
            }
            WireC2S::AuthStart { mechanism, initial } => {
                if matches!(conn, ConnAuth::Authenticated(_)) {
                    let _ = tx
                        .send(WireS2C::AuthError {
                            reason: "already authenticated".into(),
                        })
                        .await;
                } else {
                    conn = match auth.begin(&mechanism, &initial, tls_state.clone()) {
                        BeginOutcome::Challenge { data, exchange } => {
                            let _ = tx.send(WireS2C::AuthChallenge { data }).await;
                            ConnAuth::InProgress(exchange)
                        }
                        BeginOutcome::Success {
                            final_data,
                            success,
                        } => ConnAuth::Authenticated(complete_auth(&tx, final_data, success).await),
                        BeginOutcome::Failed(reject) => {
                            let _ = tx
                                .send(WireS2C::AuthError {
                                    reason: reject.reason,
                                })
                                .await;
                            ConnAuth::Unauthenticated
                        }
                    };
                }
            }
            WireC2S::AuthStep { data } => {
                match std::mem::replace(&mut conn, ConnAuth::Unauthenticated) {
                    ConnAuth::InProgress(mut exchange) => {
                        conn = match exchange.step(&data) {
                            StepOutcome::Challenge(challenge) => {
                                let _ = tx.send(WireS2C::AuthChallenge { data: challenge }).await;
                                ConnAuth::InProgress(exchange)
                            }
                            StepOutcome::Success {
                                final_data,
                                success,
                            } => ConnAuth::Authenticated(
                                complete_auth(&tx, final_data, success).await,
                            ),
                            StepOutcome::Failed(reject) => {
                                let _ = tx
                                    .send(WireS2C::AuthError {
                                        reason: reject.reason,
                                    })
                                    .await;
                                ConnAuth::Unauthenticated
                            }
                        };
                    }
                    // An AuthStep with no exchange in progress (or after auth): refuse, stay
                    // unelevated. `conn` was already reset to Unauthenticated by the take above for
                    // the non-Authenticated cases; restore Authenticated if that was the state.
                    other => {
                        conn = other;
                        let _ = tx
                            .send(WireS2C::AuthError {
                                reason: "no authentication in progress".into(),
                            })
                            .await;
                    }
                }
            }
            WireC2S::AuthResume { token } => {
                if matches!(conn, ConnAuth::Authenticated(_)) {
                    let _ = tx
                        .send(WireS2C::AuthError {
                            reason: "already authenticated".into(),
                        })
                        .await;
                } else {
                    conn = match auth.resume(&token) {
                        BeginOutcome::Success {
                            final_data,
                            success,
                        } => ConnAuth::Authenticated(complete_auth(&tx, final_data, success).await),
                        BeginOutcome::Failed(reject) => {
                            let _ = tx
                                .send(WireS2C::AuthError {
                                    reason: reject.reason,
                                })
                                .await;
                            ConnAuth::Unauthenticated
                        }
                        // `resume` never yields a challenge.
                        BeginOutcome::Challenge { .. } => ConnAuth::Unauthenticated,
                    };
                }
            }
            WireC2S::Call { id, req } => {
                if matches!(conn, ConnAuth::Authenticated(_)) {
                    let api = api.clone();
                    let tx = tx.clone();
                    // TODO(auth3-deliverable3): once Auth 2 merges, wrap this dispatch in
                    // `with_request_context(RequestContext { principal, .. })` and call
                    // `authorize(&req)` first (Forbidden on missing capability). The principal is
                    // available from the `Authenticated` arm; it is intentionally not yet threaded
                    // here because the request-context/authorize interface is owned by Track A.
                    tokio::spawn(async move {
                        let res = dispatch(api.as_ref(), req).await;
                        let _ = tx.send(WireS2C::Reply { id, res }).await;
                    });
                } else {
                    let _ = tx
                        .send(WireS2C::Reply {
                            id,
                            res: ApiResponse::Error(ApiError::Unauthenticated(
                                "authenticate before issuing requests".into(),
                            )),
                        })
                        .await;
                }
            }
            WireC2S::Open { id, req } => {
                if matches!(conn, ConnAuth::Authenticated(_)) {
                    if is_streaming(&req) {
                        streams.insert(id, spawn_stream(api.clone(), tx.clone(), id, req));
                    } else {
                        let _ = tx
                            .send(WireS2C::End {
                                id,
                                error: Some(ApiError::Unsupported(
                                    "request is not streamable; use Call".into(),
                                )),
                            })
                            .await;
                    }
                } else {
                    let _ = tx
                        .send(WireS2C::End {
                            id,
                            error: Some(ApiError::Unauthenticated(
                                "authenticate before opening a stream".into(),
                            )),
                        })
                        .await;
                }
            }
            WireC2S::Cancel { id } => {
                if let Some(handle) = streams.remove(&id) {
                    handle.abort();
                    let _ = tx.send(WireS2C::End { id, error: None }).await;
                }
            }
        }
    }
    drop(tx);
    let _ = writer.await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};
    use std::io::Write as _;
    use tempfile::TempDir;
    use tokio::net::TcpStream;
    use tokio_rustls::rustls::pki_types::{PrivatePkcs8KeyDer, ServerName};
    use tokio_rustls::rustls::ClientConfig;
    use tokio_rustls::TlsConnector;

    /// A throwaway PKI: a CA, a CA-signed server cert (SAN `localhost`), a CA-signed client cert,
    /// and an untrusted self-signed client cert.
    struct Pki {
        ca_pem: String,
        ca_der: CertificateDer<'static>,
        server_cert_pem: String,
        server_key_pem: String,
        client_cert_der: CertificateDer<'static>,
        client_key_der: PrivateKeyDer<'static>,
        bad_cert_der: CertificateDer<'static>,
        bad_key_der: PrivateKeyDer<'static>,
    }

    fn gen_pki() -> Pki {
        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).unwrap();

        let server_key = KeyPair::generate().unwrap();
        let server_cert = CertificateParams::new(vec!["localhost".to_string()])
            .unwrap()
            .signed_by(&server_key, &ca_cert, &ca_key)
            .unwrap();

        let client_key = KeyPair::generate().unwrap();
        let client_cert = CertificateParams::new(vec!["client".to_string()])
            .unwrap()
            .signed_by(&client_key, &ca_cert, &ca_key)
            .unwrap();

        let bad_key = KeyPair::generate().unwrap();
        let bad_cert = CertificateParams::new(vec!["bad".to_string()])
            .unwrap()
            .self_signed(&bad_key)
            .unwrap();

        Pki {
            ca_pem: ca_cert.pem(),
            ca_der: ca_cert.der().clone(),
            server_cert_pem: server_cert.pem(),
            server_key_pem: server_key.serialize_pem(),
            client_cert_der: client_cert.der().clone(),
            client_key_der: PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
                client_key.serialize_der(),
            )),
            bad_cert_der: bad_cert.der().clone(),
            bad_key_der: PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(bad_key.serialize_der())),
        }
    }

    fn write_file(dir: &TempDir, name: &str, contents: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path
    }

    /// Build the server config from on-disk PEM files (exercising the real [`build_server_config`]
    /// path) for the given client-auth policy.
    fn server_config(pki: &Pki, require_client_cert: bool) -> Arc<ServerConfig> {
        let dir = tempfile::tempdir().unwrap();
        let cert_path = write_file(&dir, "server.pem", &pki.server_cert_pem);
        let key_path = write_file(&dir, "server.key", &pki.server_key_pem);
        let ca_path = write_file(&dir, "ca.pem", &pki.ca_pem);
        build_server_config(&ApiTlsConfig {
            cert_path,
            key_path,
            require_client_cert,
            client_ca_path: require_client_cert.then_some(ca_path),
        })
        .expect("server config")
    }

    fn client_config(
        pki: &Pki,
        client_auth: Option<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)>,
    ) -> Arc<ClientConfig> {
        let mut roots = RootCertStore::empty();
        roots.add(pki.ca_der.clone()).unwrap();
        let provider = Arc::new(tokio_rustls::rustls::crypto::aws_lc_rs::default_provider());
        let builder = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots);
        let cfg = match client_auth {
            Some((certs, key)) => builder.with_client_auth_cert(certs, key).unwrap(),
            None => builder.with_no_client_auth(),
        };
        Arc::new(cfg)
    }

    /// Run one TLS handshake; returns whether the *server* side completed + accepted it. The server
    /// is authoritative for client-cert verification: under TLS 1.3 the client's `connect` future can
    /// resolve `Ok` before the server processes (and rejects) the client certificate, so the server
    /// accept result is the one that reflects the mTLS policy.
    async fn handshake(server: Arc<ServerConfig>, client: Arc<ClientConfig>) -> bool {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let acceptor = TlsAcceptor::from(server);
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            // tokio-rustls completes the full handshake — including client-certificate verification
            // under mTLS — before `accept` resolves, so its Ok/Err is the authoritative policy signal.
            acceptor.accept(stream).await.is_ok()
        });
        let connector = TlsConnector::from(client);
        let tcp = TcpStream::connect(addr).await.unwrap();
        let name = ServerName::try_from("localhost").unwrap();
        let _ = connector.connect(name, tcp).await;
        tokio::time::timeout(std::time::Duration::from_secs(5), server_task)
            .await
            .expect("server handshake did not settle")
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tls_handshake_succeeds_with_server_cert_no_client_auth() {
        let pki = gen_pki();
        let server = server_config(&pki, false);
        let client = client_config(&pki, None);
        assert!(
            handshake(server, client).await,
            "a client trusting the CA must complete the TLS handshake"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mtls_accepts_ca_signed_client_and_rejects_untrusted_client() {
        let pki = gen_pki();

        // A CA-signed client certificate is accepted (mTLS handshake completes).
        let server = server_config(&pki, true);
        let good_client = client_config(
            &pki,
            Some((
                vec![pki.client_cert_der.clone()],
                pki.client_key_der.clone_key(),
            )),
        );
        assert!(
            handshake(server, good_client).await,
            "a CA-signed client certificate must be accepted under mTLS"
        );

        // An untrusted (self-signed) client certificate is rejected at the TLS layer.
        let server = server_config(&pki, true);
        let bad_client = client_config(
            &pki,
            Some((vec![pki.bad_cert_der.clone()], pki.bad_key_der.clone_key())),
        );
        assert!(
            !handshake(server, bad_client).await,
            "an untrusted client certificate must be rejected during the handshake"
        );
    }

    #[test]
    fn require_client_cert_without_ca_is_an_error() {
        let pki = gen_pki();
        let dir = tempfile::tempdir().unwrap();
        let cert_path = write_file(&dir, "server.pem", &pki.server_cert_pem);
        let key_path = write_file(&dir, "server.key", &pki.server_key_pem);
        let err = build_server_config(&ApiTlsConfig {
            cert_path,
            key_path,
            require_client_cert: true,
            client_ca_path: None,
        });
        assert!(matches!(err, Err(TlsConfigError::MissingClientCa)));
    }
}
