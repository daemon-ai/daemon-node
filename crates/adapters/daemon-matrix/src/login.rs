//! The one-shot operator `matrix login` flow (spec §6.1).
//!
//! SSO is inherently interactive (it opens a browser), so it lives at bring-up, not in the headless
//! run loop. This performs the matrix-sdk SSO flow against `homeserver`, then writes the resulting
//! session blob into the credential subsystem under `credential_ref` — the same key the profile's
//! `bound_accounts` declares and that the adapter restores from at `serve` time. The crypto store is
//! created at the per-account dir keyed by `credential_ref`, so `serve` re-opens the *same* device.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use daemon_host::CredentialStore;

use crate::account::{account_store_dir, build_client, StoredSession};

/// Best-effort: open `url` in the operator's browser (the SDK also waits on its local callback
/// server). Always prints the URL so a headless/SSH operator can open it manually.
fn open_browser(url: &str) {
    println!("\nMatrix SSO — open this URL in a browser to log in:\n  {url}\n");
    // Try the common openers; ignore failures (the URL is already printed).
    for opener in ["xdg-open", "open"] {
        if std::process::Command::new(opener).arg(url).spawn().is_ok() {
            break;
        }
    }
}

/// Run the SSO login for one account and persist its session under `credential_ref`.
///
/// `store_root` is the absolute per-account store root (`<data_dir>/<matrix.store_root>`); the
/// account's state + crypto store is created at `<store_root>/<credential_ref>/`.
pub async fn login(
    store: Arc<dyn CredentialStore>,
    homeserver: &str,
    store_root: &Path,
    credential_ref: &str,
) -> Result<()> {
    let store_dir = account_store_dir(store_root, credential_ref);
    let client = build_client(homeserver, &store_dir).await?;

    client
        .matrix_auth()
        .login_sso(|url| async move {
            open_browser(&url);
            Ok(())
        })
        .initial_device_display_name("daemon")
        .await
        .map_err(|e| anyhow!("matrix SSO login failed: {e}"))?;

    let session = client
        .matrix_auth()
        .session()
        .ok_or_else(|| anyhow!("no session present after SSO login"))?;
    let user_id = client
        .user_id()
        .ok_or_else(|| anyhow!("client has no user id after login"))?
        .to_owned();

    let stored = StoredSession {
        homeserver: homeserver.to_string(),
        session,
    };
    store
        .set(credential_ref, &stored.to_blob()?)
        .map_err(|e| anyhow!("writing matrix session to credential store: {e}"))
        .context("persisting matrix session")?;

    println!(
        "matrix: logged in as {user_id}; session stored under credential-ref `{credential_ref}`.\n\
         Bind it to a profile via `bound_accounts` with transport_instance `matrix/{user_id}`."
    );
    Ok(())
}
