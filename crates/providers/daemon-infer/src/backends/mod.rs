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
/// Offline GGUF quantization (llama.cpp's native quantizer); only built with the `llama` feature.
#[cfg(feature = "llama")]
pub mod quantize;

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

/// Whether the linked llama.cpp build was compiled with a GPU backend (CUDA / Vulkan / Metal / …).
///
/// This reflects the *build*, not the presence of a runtime device: a Vulkan-enabled `libllama`
/// returns `true` even before any device is probed. Used by the engine lane's integration tests to
/// assert the worker is actually a GPU build (vs. a silent CPU-only fallback). Only meaningful with
/// the `llama` feature; the stub build reports `false`.
#[cfg(feature = "llama")]
#[must_use]
pub fn gpu_offload_supported() -> bool {
    llama_cpp_4::supports_gpu_offload()
}

/// llama.cpp's system/build info string (CPU features + compiled backends). Diagnostic only.
#[cfg(feature = "llama")]
#[must_use]
pub fn system_info() -> String {
    llama_cpp_4::print_system_info()
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
