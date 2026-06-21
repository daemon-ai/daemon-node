//! Offline GGUF quantization via llama.cpp's native quantizer.
//!
//! This is the only place that links `llama-cpp-4`'s `model_quantize` (PR-grade `llama-quantize`
//! equivalent). The daemon's `daemon-models` crate shells out to the worker's `quantize` subcommand
//! rather than linking the engine itself, so quant kernels are never reimplemented and the heavy ML
//! tree stays isolated to this feature-gated worker.

use llama_cpp_4::quantize::{LlamaFtype, QuantizeParams};

/// Quantize the GGUF at `input` to `output` at the `ftype` precision (e.g. `"Q4_K_M"`).
///
/// `nthread` of `0` lets llama.cpp auto-detect. Returns a human-readable error string on an unknown
/// ftype or a non-zero quantizer return code.
pub fn run_quantize(input: &str, output: &str, ftype: &str, nthread: i32) -> Result<(), String> {
    let parsed = LlamaFtype::from_name(ftype).ok_or_else(|| {
        let known: Vec<&str> = LlamaFtype::all().iter().map(|f| f.name()).collect();
        format!("unknown quant type '{ftype}'; known: {}", known.join(", "))
    })?;
    let mut params = QuantizeParams::new(parsed);
    if nthread > 0 {
        params = params.with_nthread(nthread);
    }
    llama_cpp_4::model_quantize(input, output, &params)
        .map_err(|rc| format!("llama.cpp model_quantize failed (rc={rc})"))
}
