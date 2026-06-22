//! The one-shot operator `matrix login` flow (spec §6.1).
//!
//! SSO is inherently interactive (it opens a browser), so it lives at bring-up, not in the headless
//! run loop. This is the *operator* path; it shares the exact begin/complete primitives the wire
//! `AuthApi` family uses ([`crate::sso_begin`] / [`crate::sso_complete`]), differing only in how the
//! redirect is captured: here a tiny local loopback HTTP listener stands in for the GUI's browser +
//! redirect capture. The resulting session blob is written into the credential subsystem under
//! `credential_ref` — the same key the profile's `bound_accounts` declares and that `serve` restores
//! from, with the on-disk crypto store keyed identically so the same device is re-opened (spec §6.3).

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use daemon_host::CredentialStore;

use crate::auth::{sso_begin, sso_complete};

/// Best-effort: open `url` in the operator's browser. Always prints the URL so a headless/SSH operator
/// can open it manually.
fn open_browser(url: &str) {
    println!("\nMatrix SSO — open this URL in a browser to log in:\n  {url}\n");
    for opener in ["xdg-open", "open"] {
        if std::process::Command::new(opener).arg(url).spawn().is_ok() {
            break;
        }
    }
}

/// The minimal HTML returned to the browser once the redirect is captured.
const DONE_PAGE: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n\
     <!doctype html><meta charset=utf-8><title>daemon</title>\
     <body style=\"font-family:sans-serif\"><h2>Login complete</h2>\
     <p>You can close this tab and return to the terminal.</p></body>";

/// Accept one redirect on the loopback listener and return its request target (`/?loginToken=…`).
async fn capture_redirect(listener: TcpListener) -> Result<String> {
    let (mut stream, _) = listener
        .accept()
        .await
        .context("accepting the SSO redirect on the loopback listener")?;

    // Read the request head; the request line (`GET <target> HTTP/1.1`) carries the loginToken query.
    let mut buf = [0u8; 4096];
    let n = stream
        .read(&mut buf)
        .await
        .context("reading the SSO redirect request")?;
    let head = String::from_utf8_lossy(&buf[..n]);
    let target = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| anyhow!("malformed SSO redirect request"))?
        .to_string();

    let _ = stream.write_all(DONE_PAGE.as_bytes()).await;
    let _ = stream.shutdown().await;
    Ok(target)
}

/// Run the SSO login for one account and persist its session under `credential_ref`.
///
/// `store_root` is the absolute per-account store root (`<data_dir>/<matrix.store_root>`); the
/// account's state + crypto store is created at `<store_root>/<credential_ref>/`. The redirect is
/// captured by a local loopback listener (the operator analogue of the GUI's browser hop), then the
/// flow is finished through the shared [`sso_complete`] primitive.
pub async fn login(
    store: Arc<dyn CredentialStore>,
    homeserver: &str,
    store_root: &Path,
    credential_ref: &str,
) -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("binding the loopback SSO redirect listener")?;
    let port = listener
        .local_addr()
        .context("reading the loopback listener address")?
        .port();
    let redirect_uri = format!("http://127.0.0.1:{port}/");

    let session = sso_begin(store_root, homeserver, credential_ref, &redirect_uri, None).await?;
    open_browser(&session.authorization_url);

    let callback = capture_redirect(listener).await?;
    let login = sso_complete(session, &callback).await?;

    store
        .set(&login.credential_ref, &login.credential_blob)
        .map_err(|e| anyhow!("writing matrix session to credential store: {e}"))
        .context("persisting matrix session")?;

    println!(
        "matrix: logged in as {user}; session stored under credential-ref `{cref}`.\n\
         Bind it to a profile via `bound_accounts` with transport_instance `{instance}`.",
        user = login.user_id,
        cref = login.credential_ref,
        instance = login.transport_instance.as_str(),
    );
    Ok(())
}
