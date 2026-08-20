// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The TLS TCP transport for the node api surface (deliverable (2) of the Auth 3 track).
//!
//! The Unix-domain socket ([`crate::socket`]) stays plaintext (local trust); any *networked* api
//! access goes over TLS and **always requires authentication**. [`serve_api_tls_tcp`] accepts TCP
//! connections, performs the `rustls` handshake (server cert always; client cert verified when
//! `require_client_cert` is set, i.e. mTLS), captures the peer-certificate fingerprint for the
//! EXTERNAL mechanism, and then hands the connection to the shared, context-aware multiplexed loop
//! [`crate::socket::serve_mux`] in [`crate::socket::AuthMode::Required`] mode: a connection must
//! complete a SASL exchange before any `Call`/`Open` is served (pre-auth resolves to
//! [`daemon_api::ApiError::Unauthenticated`] and the connection stays unelevated), and once
//! authenticated every dispatch runs under the principal's [`crate::request_context::RequestContext`]
//! through the capability gate ([`crate::authz::authorize`]).
//!
//! Crypto provider: this module pins the **aws-lc-rs** `rustls` provider, matching the provider the
//! rest of the dependency tree already resolves (`cargo tree -i rustls` -> rustls 0.23 + aws-lc-rs),
//! so no second crypto backend is introduced.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use daemon_api::NodeApi;
use daemon_common::{IngressGovernor, PeerKey};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio_rustls::rustls::client::danger::HandshakeSignatureValid;
use tokio_rustls::rustls::crypto::CryptoProvider;
use tokio_rustls::rustls::pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, UnixTime};
use tokio_rustls::rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use tokio_rustls::rustls::server::WebPkiClientVerifier;
use tokio_rustls::rustls::{
    DigitallySignedStruct, DistinguishedName, RootCertStore, ServerConfig, SignatureScheme,
};
use tokio_rustls::TlsAcceptor;

use crate::authn::{Authenticator, TlsState};
use crate::socket::{next_conn_id, serve_mux, AuthMode};

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
    /// A PEM chain parsed but contained no certificates (leaf fingerprint needs one).
    #[error("{0}: no certificates in PEM chain")]
    EmptyChain(String),
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

/// The **request-optional-any** client-certificate verifier (pairing spec §5.1), the default
/// client-auth mode of the TLS api listener: the server *requests* a client certificate but does
/// not require one, and *any* presented certificate passes the TLS layer — after its handshake
/// signature is verified, so possession of the matching private key is still proven. All trust
/// judgment is deferred to SASL, which is fail-closed: an unmapped fingerprint means EXTERNAL
/// denies, and SCRAM/PLAIN are unaffected. The only new capability over `with_no_client_auth` is
/// that a presented self-signed certificate reaches [`TlsState::peer_cert_fingerprint`] — which
/// pairing enrollment and EXTERNAL need.
#[derive(Debug)]
struct AcceptAnyClientCert {
    provider: Arc<CryptoProvider>,
    /// Empty: we hint no CA subjects, so clients offer whatever identity they have.
    subjects: Vec<DistinguishedName>,
}

impl AcceptAnyClientCert {
    fn new(provider: Arc<CryptoProvider>) -> Self {
        Self {
            provider,
            subjects: Vec::new(),
        }
    }
}

impl ClientCertVerifier for AcceptAnyClientCert {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &self.subjects
    }

    fn verify_client_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, tokio_rustls::rustls::Error> {
        // Any certificate is admitted at the TLS layer; SASL owns the trust decision.
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, tokio_rustls::rustls::Error> {
        tokio_rustls::rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, tokio_rustls::rustls::Error> {
        tokio_rustls::rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }

    fn offer_client_auth(&self) -> bool {
        true
    }

    fn client_auth_mandatory(&self) -> bool {
        false
    }
}

/// Build a rustls [`ServerConfig`] from the resolved [`ApiTlsConfig`], pinning the aws-lc-rs crypto
/// provider (matching the rest of the tree). With `require_client_cert`, an mTLS verifier is built
/// over the configured client-CA bundle so untrusted client certificates are rejected during the
/// handshake. Otherwise (the default) client certificates are **requested but optional**, and any
/// presented certificate is admitted at the TLS layer ([`AcceptAnyClientCert`], pairing spec §5.1):
/// clients presenting nothing still connect and SCRAM as before, while a presented (e.g.
/// self-signed) certificate reaches [`TlsState::peer_cert_fingerprint`] for EXTERNAL/pairing —
/// where trust is decided, fail-closed.
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
        builder
            .with_client_cert_verifier(Arc::new(AcceptAnyClientCert::new(provider)))
            .with_single_cert(certs, key)?
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

/// The SHA-256 fingerprint (lowercase hex) of the FIRST certificate in a PEM chain file —
/// byte-for-byte the format of [`TlsState::peer_cert_fingerprint`] and of the app's
/// `conn/tls/pinnedSha256` key, so an accepted fingerprint drops in with no transformation.
/// Published as the discovery TXT `fp` hint (daemon-lan-discovery-spec.md §2.2) — an untrusted
/// hint app-side; never a trust decision input node-side.
pub fn leaf_cert_fingerprint(path: &Path) -> Result<String, TlsConfigError> {
    let certs = load_certs(path)?;
    let leaf = certs
        .first()
        .ok_or_else(|| TlsConfigError::EmptyChain(path.display().to_string()))?;
    Ok(hex(&Sha256::digest(leaf.as_ref())))
}

/// The SHA-256 fingerprint (hex) of the peer's leaf certificate, if one was presented + verified.
fn peer_fingerprint<IO>(stream: &tokio_rustls::server::TlsStream<IO>) -> Option<String> {
    let (_, conn) = stream.get_ref();
    conn.peer_certificates()
        .and_then(|certs| certs.first())
        .map(|cert| hex(&Sha256::digest(cert.as_ref())))
}

/// Serve the node api surface over TLS/TCP until the listener errors. Every connection is mux-only
/// and **must authenticate** (TCP is never local-trusted): the connection is handed to the shared
/// [`serve_mux`] in [`AuthMode::Required`] mode with the per-connection [`TlsState`] (carrying the
/// verified client-cert fingerprint for EXTERNAL). Spawn it as a background task alongside the Unix
/// listener.
pub async fn serve_api_tls_tcp(
    listener: TcpListener,
    tls: Arc<ServerConfig>,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
    governor: Arc<IngressGovernor>,
) {
    let acceptor = TlsAcceptor::from(tls);
    let limits = governor.limits();
    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                // Cluster F: per-peer connection rate + global concurrency, both fail-closed, BEFORE
                // the (costly) TLS handshake is spawned. A refused connection is dropped cleanly.
                let peer = PeerKey::ip(addr.ip());
                if let Err(e) = governor.check_peer(&peer) {
                    tracing::debug!(%addr, "tls connection refused: {e}");
                    continue;
                }
                let Some(permit) = governor.admit_connection() else {
                    tracing::debug!(%addr, "tls connection refused: connection cap reached");
                    continue;
                };
                let acceptor = acceptor.clone();
                let api = api.clone();
                let auth = auth.clone();
                tokio::spawn(async move {
                    // Hold the connection permit for the whole connection; RAII-dropped on return
                    // (incl. the secret-epoch revocation teardown path), releasing the slot.
                    let _permit = permit;
                    match acceptor.accept(stream).await {
                        Ok(tls_stream) => {
                            let tls_state = TlsState {
                                is_tls: true,
                                peer_cert_fingerprint: peer_fingerprint(&tls_stream),
                            };
                            let mode = Arc::new(AuthMode::Required { auth, tls_state });
                            let (rd, wr) = tokio::io::split(tls_stream);
                            if let Err(e) =
                                serve_mux(rd, wr, api, mode, None, next_conn_id(), limits).await
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

    /// Like [`handshake`], but also returns the server-observed peer-certificate fingerprint of
    /// an accepted connection (`None` = handshake rejected; `Some(None)` = accepted, no client
    /// cert presented; `Some(Some(fp))` = accepted with a captured fingerprint).
    async fn handshake_fp(
        server: Arc<ServerConfig>,
        client: Arc<ClientConfig>,
    ) -> Option<Option<String>> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let acceptor = TlsAcceptor::from(server);
        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            acceptor
                .accept(stream)
                .await
                .ok()
                .map(|s| peer_fingerprint(&s))
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

    /// The §5.1 verifier-mode matrix, default (request-optional-any) side: presenting nothing
    /// still connects exactly as before (no fingerprint), and a presented **self-signed**
    /// certificate is admitted at the TLS layer with its fingerprint captured for SASL — where
    /// trust is decided fail-closed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn optional_client_auth_matrix_captures_any_presented_cert() {
        let pki = gen_pki();

        // No client certificate: connects, nothing captured (the pre-§5.1 behavior, preserved).
        let server = server_config(&pki, false);
        let client = client_config(&pki, None);
        assert_eq!(
            handshake_fp(server, client).await,
            Some(None),
            "a certificate-less client must still connect under request-optional-any"
        );

        // A self-signed client certificate: admitted, and the captured fingerprint is the
        // SHA-256 of its DER — the pin/TXT-fp/EXTERNAL format.
        let server = server_config(&pki, false);
        let client = client_config(
            &pki,
            Some((vec![pki.bad_cert_der.clone()], pki.bad_key_der.clone_key())),
        );
        let expected = hex(&Sha256::digest(pki.bad_cert_der.as_ref()));
        assert_eq!(
            handshake_fp(server, client).await,
            Some(Some(expected)),
            "a self-signed client certificate must be admitted and fingerprinted"
        );

        // A CA-signed client certificate is equally admitted (any presented cert passes TLS).
        let server = server_config(&pki, false);
        let client = client_config(
            &pki,
            Some((
                vec![pki.client_cert_der.clone()],
                pki.client_key_der.clone_key(),
            )),
        );
        let expected = hex(&Sha256::digest(pki.client_cert_der.as_ref()));
        assert_eq!(handshake_fp(server, client).await, Some(Some(expected)));
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
