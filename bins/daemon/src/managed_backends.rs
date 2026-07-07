// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The binary-side [`daemon_host::ManagedResource`] implementations: the node-managed backend
//! resources reported in `HealthReport.services` alongside the resident-service supervisor.
//!
//! - [`GatewayResource`] promotes the OpenAI-compatible gateway from a boot-only `tokio::spawn` into
//!   a resident, wire-configurable service: it owns the bound listener's lifecycle, persists its
//!   enable/rebind state to the durable store (boot `[gateway]` config is the default/fallback),
//!   hot-(re)binds on `GatewaySet`, and reports addr/listening/last_error as the `"gateway"` health
//!   line.
//! - [`LocalInferenceResource`] surfaces the local-inference worker's lifted status
//!   ([`LocalInferenceStatus`]) as the `"local-inference"` health line — a pure observation seam that
//!   changes none of the provider's recovery behavior.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{ApiError, GatewayStatus, ServiceHealth};
use daemon_gateway::GatewayBackend;
use daemon_host::{GatewayControl, ManagedResource, ServiceError};
use daemon_providers::{LocalInferenceState, LocalInferenceStatus};
use daemon_store::SessionStore;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// The node-owned OpenAI-compatible gateway as a resident, wire-configurable managed resource.
pub struct GatewayResource {
    /// The gateway request backend (node catalog + provider resolution), shared across (re)binds.
    backend: Arc<dyn GatewayBackend>,
    /// The bearer token clients present (minted/pinned at boot).
    token: String,
    /// The durable store the enable/rebind override persists to (boot config is the fallback).
    store: Arc<dyn SessionStore>,
    /// The boot `[gateway].addr` — the default address when no runtime override is set.
    boot_addr: Option<String>,
    /// The boot enable state (`true` iff `[gateway].addr` was configured).
    boot_enabled: bool,
    /// The live listener/task state, guarded by an async mutex (held across the bind await).
    rt: Mutex<GatewayRuntime>,
}

/// The gateway's live runtime state.
#[derive(Default)]
struct GatewayRuntime {
    /// The running serve task (aborted on rebind/stop).
    task: Option<JoinHandle<()>>,
    /// Whether the gateway is configured to serve.
    enabled: bool,
    /// The effective (last-applied) bind address.
    addr: Option<String>,
    /// Whether the listener is currently bound.
    listening: bool,
    /// The last bind/serve error, if the most recent (re)bind failed.
    last_error: Option<String>,
    /// Successful listener binds so far (the reported `restarts` are binds beyond the first).
    binds: u32,
}

impl GatewayResource {
    /// Build the resource over the gateway backend + bearer token, the durable store the override
    /// persists to, and the boot `[gateway]` defaults. The listener is bound lazily by
    /// [`ManagedResource::activate`] (called once at node startup).
    pub fn new(
        backend: Arc<dyn GatewayBackend>,
        token: String,
        store: Arc<dyn SessionStore>,
        boot_addr: Option<String>,
        boot_enabled: bool,
    ) -> Self {
        Self {
            backend,
            token,
            store,
            boot_addr,
            boot_enabled,
            rt: Mutex::new(GatewayRuntime::default()),
        }
    }

    /// Resolve the effective `(enabled, addr)` — the durable runtime override layered on top of the
    /// boot config (an override addr of `None` falls back to the boot addr).
    async fn resolve_effective(&self) -> (bool, Option<String>) {
        match self.store.gateway_override().await {
            Some((enabled, addr)) => (enabled, addr.or_else(|| self.boot_addr.clone())),
            None => (self.boot_enabled, self.boot_addr.clone()),
        }
    }

    /// Apply an `(enabled, addr)` desired state to the live listener: abort any running serve task,
    /// then (when enabled with an address) bind + serve, recording listening/last_error. Idempotent
    /// and best-effort — a bind failure is recorded, not propagated, so health reflects it.
    async fn apply(&self, enabled: bool, addr: Option<String>) {
        let mut rt = self.rt.lock().await;
        if let Some(task) = rt.task.take() {
            task.abort();
        }
        rt.enabled = enabled;
        rt.addr = addr.clone();
        rt.listening = false;
        rt.last_error = None;
        if !enabled {
            return;
        }
        let Some(addr) = addr else {
            rt.last_error = Some("gateway enabled but no bind address configured".into());
            return;
        };
        match TcpListener::bind(&addr).await {
            Ok(listener) => {
                let backend = self.backend.clone();
                let token = self.token.clone();
                tracing::info!(
                    %addr,
                    "serving OpenAI-compatible gateway (POST /v1/chat/completions + GET /v1/models, bearer-gated)"
                );
                let task = tokio::spawn(async move {
                    if let Err(e) = daemon_gateway::serve(listener, backend, token).await {
                        tracing::warn!(error = %e, "gateway surface ended");
                    }
                });
                rt.task = Some(task);
                rt.listening = true;
                rt.binds += 1;
            }
            Err(e) => {
                rt.last_error = Some(format!("bind {addr}: {e}"));
                tracing::warn!(%addr, error = %e, "gateway bind failed");
            }
        }
    }

    /// Build the wire status from the current runtime state.
    fn status_of(rt: &GatewayRuntime) -> GatewayStatus {
        GatewayStatus {
            enabled: rt.enabled,
            addr: rt.addr.clone(),
            listening: rt.listening,
            last_error: rt.last_error.clone(),
        }
    }
}

#[async_trait]
impl ManagedResource for GatewayResource {
    fn name(&self) -> &str {
        "gateway"
    }

    async fn activate(&self) -> Result<(), ServiceError> {
        let (enabled, addr) = self.resolve_effective().await;
        self.apply(enabled, addr).await;
        // A bind failure is surfaced to the caller (which logs it) but never aborts node startup —
        // the resource stays resident and its health reports the error.
        let rt = self.rt.lock().await;
        if rt.enabled && !rt.listening {
            return Err(ServiceError::new(
                rt.last_error
                    .clone()
                    .unwrap_or_else(|| "gateway failed to bind".into()),
            ));
        }
        Ok(())
    }

    async fn health(&self) -> ServiceHealth {
        let rt = self.rt.lock().await;
        // A disabled gateway is intentionally not serving (ok); an enabled one is ok only while its
        // listener is bound. `detail` carries the addr/listening/last_error triad.
        let ok = !rt.enabled || rt.listening;
        let detail = serde_json::json!({
            "addr": rt.addr,
            "listening": rt.listening,
            "last_error": rt.last_error,
        })
        .to_string();
        ServiceHealth {
            name: "gateway".to_string(),
            ok,
            restarts: rt.binds.saturating_sub(1),
            detail: Some(detail),
        }
    }

    async fn stop(&self) {
        let mut rt = self.rt.lock().await;
        if let Some(task) = rt.task.take() {
            task.abort();
        }
        rt.listening = false;
    }
}

#[async_trait]
impl GatewayControl for GatewayResource {
    async fn get(&self) -> GatewayStatus {
        Self::status_of(&*self.rt.lock().await)
    }

    async fn set(&self, enabled: bool, addr: Option<String>) -> Result<GatewayStatus, ApiError> {
        // An explicit addr wins; otherwise keep the current effective addr (falling back to boot),
        // so a bare enable/disable never loses the address. The resolved concrete addr is persisted
        // so a restart re-binds it.
        let current_addr = self.rt.lock().await.addr.clone();
        let new_addr = addr.or(current_addr).or_else(|| self.boot_addr.clone());
        self.store
            .set_gateway_override(enabled, new_addr.as_deref())
            .await
            .map_err(|e| ApiError::Other(format!("persist gateway override: {e}")))?;
        self.apply(enabled, new_addr).await;
        Ok(self.get().await)
    }
}

/// The node's local-inference worker surfaced as a managed resource: it reads the lifted
/// [`LocalInferenceStatus`] handle and reports `idle | loading | loaded | crashed` as the
/// `"local-inference"` health line. The provider owns its own lazy-spawn / respawn / meltdown
/// recovery — this resource only observes it (`activate`/`stop` are no-ops).
pub struct LocalInferenceResource {
    status: LocalInferenceStatus,
}

impl LocalInferenceResource {
    /// Build the resource over the shared worker status handle.
    pub fn new(status: LocalInferenceStatus) -> Self {
        Self { status }
    }
}

#[async_trait]
impl ManagedResource for LocalInferenceResource {
    fn name(&self) -> &str {
        "local-inference"
    }

    async fn activate(&self) -> Result<(), ServiceError> {
        // The worker is spawned lazily on first use by the provider; nothing to bring up here.
        Ok(())
    }

    async fn health(&self) -> ServiceHealth {
        let (ok, detail) = match self.status.state() {
            LocalInferenceState::Idle => (true, "idle"),
            LocalInferenceState::Loading => (true, "loading"),
            LocalInferenceState::Loaded => (true, "loaded"),
            LocalInferenceState::Crashed => (false, "crashed"),
        };
        ServiceHealth {
            name: "local-inference".to_string(),
            ok,
            restarts: self.status.restarts(),
            detail: Some(detail.to_string()),
        }
    }

    async fn stop(&self) {
        // The worker is torn down with the provider; nothing to stop here.
    }
}
