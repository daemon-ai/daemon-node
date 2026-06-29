// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Operator-readable rendering of [`daemon_api::ApiResponse`].
//!
//! The response surface is large (~60 variants), so rendering is partitioned by surface: each
//! `try_render` handles its own variants (printing them, returning `None`) and hands anything it
//! does not own back as `Some(resp)`, so [`render`] is a thin chain — mirroring the `serve_*`
//! fan-out in `daemon-api`'s `dispatch`. The final [`misc::render_rest`] is total (it carries the
//! `{:?}` catch-all), so the compiler still proves every variant is handled across the chain.

use daemon_api::ApiResponse;

mod cron;
mod curator;
mod general;
mod messaging;
mod misc;
mod model;
mod orchestration;
mod profile;
mod session;

/// Render an api response in a compact, operator-readable form.
pub(crate) fn render(resp: ApiResponse) {
    let Some(resp) = general::try_render(resp) else {
        return;
    };
    let Some(resp) = session::try_render(resp) else {
        return;
    };
    let Some(resp) = orchestration::try_render(resp) else {
        return;
    };
    let Some(resp) = model::try_render(resp) else {
        return;
    };
    let Some(resp) = profile::try_render(resp) else {
        return;
    };
    let Some(resp) = curator::try_render(resp) else {
        return;
    };
    let Some(resp) = cron::try_render(resp) else {
        return;
    };
    let Some(resp) = messaging::try_render(resp) else {
        return;
    };
    misc::render_rest(resp);
}
