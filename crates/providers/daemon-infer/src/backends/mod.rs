//! Backend selection: map a [`protocol::Engine`] + load params to a live [`InferenceBackend`].
//!
//! Each engine impl lives behind its feature (`llama`, `mistralrs`) and is `cfg`-gated, so the
//! default workspace build compiles only the [`StubBackend`] fallback (no cmake / no ML tree). The
//! daemon-side `LocalProvider` is engine-agnostic; only [`load`] here knows which crate to call.

use crate::backend::{BackendError, InferenceBackend};
use crate::protocol::{Engine, ModelParams};

#[cfg(feature = "llama")]
mod llama;
#[cfg(feature = "mistralrs")]
mod mistralrs;

/// Load `model` into a backend for `engine`. Returns a classified [`BackendError`] when the
/// requested engine was not compiled into this worker build.
pub async fn load(
    engine: Engine,
    model: &str,
    params: &ModelParams,
) -> Result<Box<dyn InferenceBackend>, BackendError> {
    match engine {
        Engine::Llama => {
            #[cfg(feature = "llama")]
            {
                let backend = llama::LlamaCppBackend::load(model, params)?;
                Ok(Box::new(backend))
            }
            #[cfg(not(feature = "llama"))]
            {
                let _ = (model, params);
                Err(BackendError::fatal(
                    "llama backend not compiled (rebuild daemon-infer with --features llama)",
                ))
            }
        }
        Engine::MistralRs => {
            #[cfg(feature = "mistralrs")]
            {
                let backend = mistralrs::MistralRsBackend::load(model, params).await?;
                Ok(Box::new(backend))
            }
            #[cfg(not(feature = "mistralrs"))]
            {
                let _ = (model, params);
                Err(BackendError::fatal(
                    "mistralrs backend not compiled (rebuild daemon-infer with --features mistralrs)",
                ))
            }
        }
    }
}

/// The identifier of the engine compiled for `engine` (for [`crate::protocol::Event::Health`]).
pub fn backend_name(engine: Engine) -> &'static str {
    match engine {
        Engine::Llama => {
            if cfg!(feature = "llama") {
                "llama"
            } else {
                "stub"
            }
        }
        Engine::MistralRs => {
            if cfg!(feature = "mistralrs") {
                "mistralrs"
            } else {
                "stub"
            }
        }
    }
}
