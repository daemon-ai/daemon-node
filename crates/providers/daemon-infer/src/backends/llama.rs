//! The llama.cpp backend (`llama-cpp-4`).
//!
//! `llama-cpp-4` is synchronous and `!Send` (`LlamaModel`/`LlamaContext` wrap raw pointers), so the
//! model is loaded and used on a single dedicated OS thread. [`LlamaCppBackend`] holds an mpsc sender
//! to that thread; each [`InferenceBackend::generate`] enqueues a job (request + chunk sender + a
//! cancel token + a oneshot for the terminal result) and awaits its completion. tokio's
//! `UnboundedSender` and `tokio_util`'s `CancellationToken` are both usable from the sync thread, so
//! the thread streams chunks and polls cancellation without any async runtime of its own.
//!
//! Phase 1 streams text + usage + classified errors + cooperative cancel. Native tool-call parsing
//! (this engine emits none) lands in the Phase-1b tool-calling pass; until then `supports_native_tools`
//! is false and no tool chunks are produced.

use std::num::NonZeroU32;
use std::sync::mpsc as std_mpsc;

use llama_cpp_4::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_4::context::LlamaContext;
use llama_cpp_4::llama_backend::LlamaBackend;
use llama_cpp_4::llama_batch::LlamaBatch;
use llama_cpp_4::model::params::LlamaModelParams;
use llama_cpp_4::model::{AddBos, LlamaChatMessage, LlamaModel, Special};
use llama_cpp_4::sampling::LlamaSampler;
use llama_cpp_4::token::LlamaToken;
use llama_cpp_4::DecodeError;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendChunk, BackendError, GenerateRequest, InferenceBackend};
use crate::protocol::{Capabilities, Constraint, ModelParams, Sampling, ToolCallFormat, Usage};
use crate::tooling;

/// The output-token cap when a request leaves `max_tokens` unset (`0`).
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// A persistent generation context reused across [`Task::Generate`] jobs for prompt caching.
///
/// `llama-cpp-4` keeps the KV cache inside a [`LlamaContext`]; the prior design rebuilt the context
/// every turn, discarding all prefill work. Holding the context alive — together with the exact
/// token sequence currently materialized in sequence `0`'s KV (`cached_tokens`) — lets a new request
/// reuse the longest common prefix and re-decode only the divergent suffix (see [`run_generation`]).
///
/// Single-sequence (`seq_id 0`) for now: interleaving two unrelated conversations on one worker
/// reuses only their shared leading prefix (typically system prompt + tools). Per-conversation slots
/// are a future enhancement. The cached layout depends on the context params (`n_ctx`, threads,
/// flash-attention), so a change to any of those rebuilds the session from scratch.
struct GenSession<'a> {
    /// The live context owning the KV cache (borrows the worker thread's loaded model).
    ctx: LlamaContext<'a>,
    /// The exact tokens currently held in sequence `0`'s KV cache (prompt + tokens generated last
    /// turn), against which the next request's prompt is prefix-matched.
    cached_tokens: Vec<LlamaToken>,
    /// The context window this session was built with (a change invalidates the KV layout).
    n_ctx: u32,
    /// The generation thread count this session was built with.
    n_threads: Option<u32>,
    /// Whether flash-attention was enabled for this session.
    flash_attn: bool,
}

/// The length of the longest shared leading run of `a` and `b` (the reusable KV-cache prefix).
fn find_common_prefix<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// One generation handed to the dedicated llama thread.
struct Job {
    req: GenerateRequest,
    chunks: UnboundedSender<BackendChunk>,
    cancel: CancellationToken,
    done: oneshot::Sender<Result<Usage, BackendError>>,
}

/// One embed request handed to the dedicated llama thread.
struct EmbedJob {
    texts: Vec<String>,
    done: oneshot::Sender<Result<Vec<Vec<f32>>, BackendError>>,
}

/// Work for the dedicated llama thread: a streaming generation or a batched embed.
enum Task {
    Generate(Job),
    Embed(EmbedJob),
}

/// A loaded llama.cpp model whose generations/embeds run on a dedicated thread.
pub struct LlamaCppBackend {
    jobs: std_mpsc::Sender<Task>,
    capabilities: Capabilities,
}

impl LlamaCppBackend {
    /// Spawn the engine thread, load `model_path`, and block until it reports ready (or fails).
    pub fn load(model_path: &str, params: &ModelParams) -> Result<Self, BackendError> {
        let (jobs_tx, jobs_rx) = std_mpsc::channel::<Task>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<Capabilities, BackendError>>();
        let model_path = model_path.to_string();
        let params = params.clone();
        std::thread::Builder::new()
            .name("llama-infer".into())
            .spawn(move || worker_thread(model_path, params, jobs_rx, ready_tx))
            .map_err(|e| BackendError::fatal(format!("spawn llama thread: {e}")))?;
        match ready_rx.recv() {
            Ok(Ok(capabilities)) => Ok(Self {
                jobs: jobs_tx,
                capabilities,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(BackendError::fatal(
                "llama worker thread exited during load",
            )),
        }
    }
}

#[async_trait::async_trait]
impl InferenceBackend for LlamaCppBackend {
    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    async fn generate(
        &self,
        req: GenerateRequest,
        tx: UnboundedSender<BackendChunk>,
        cancel: CancellationToken,
    ) -> Result<Usage, BackendError> {
        let (done_tx, done_rx) = oneshot::channel();
        let job = Job {
            req,
            chunks: tx,
            cancel,
            done: done_tx,
        };
        self.jobs
            .send(Task::Generate(job))
            .map_err(|_| BackendError::fatal("llama worker thread is gone"))?;
        done_rx
            .await
            .map_err(|_| BackendError::transient("llama worker dropped the job"))?
    }

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BackendError> {
        let (done_tx, done_rx) = oneshot::channel();
        self.jobs
            .send(Task::Embed(EmbedJob {
                texts,
                done: done_tx,
            }))
            .map_err(|_| BackendError::fatal("llama worker thread is gone"))?;
        done_rx
            .await
            .map_err(|_| BackendError::transient("llama worker dropped the embed job"))?
    }
}

/// The dedicated thread: init the backend, load the model, then service jobs until the channel closes.
fn worker_thread(
    model_path: String,
    params: ModelParams,
    jobs_rx: std_mpsc::Receiver<Task>,
    ready_tx: std_mpsc::Sender<Result<Capabilities, BackendError>>,
) {
    let backend = match LlamaBackend::init() {
        Ok(backend) => backend,
        Err(e) => {
            let _ = ready_tx.send(Err(BackendError::fatal(format!("llama backend init: {e}"))));
            return;
        }
    };

    let mut model_params = LlamaModelParams::default();
    if params.n_gpu_layers > 0 {
        model_params = model_params.with_n_gpu_layers(params.n_gpu_layers);
    }
    let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params) {
        Ok(model) => model,
        Err(e) => {
            let _ = ready_tx.send(Err(BackendError::fatal(format!(
                "load model {model_path}: {e}"
            ))));
            return;
        }
    };

    let capabilities = Capabilities {
        supports_native_tools: false,
        supports_streaming: true,
        // The engine emits no native tool calls; Phase 1b parses text per this format.
        tool_call_format: ToolCallFormat::HermesXml,
        max_context: Some(model.n_ctx_train()),
    };
    if ready_tx.send(Ok(capabilities)).is_err() {
        return;
    }
    drop(ready_tx);

    // The persistent generation session (KV cache + its token sequence), reused across Generate
    // jobs so a growing conversation re-decodes only its new suffix instead of the full prompt.
    let mut session: Option<GenSession> = None;

    while let Ok(task) = jobs_rx.recv() {
        match task {
            Task::Generate(job) => {
                let result = run_generation(
                    &backend,
                    &model,
                    &params,
                    &job.req,
                    &job.chunks,
                    &job.cancel,
                    &mut session,
                );
                let _ = job.done.send(result);
            }
            Task::Embed(job) => {
                let result = run_embed(&backend, &model, &params, &job.texts);
                let _ = job.done.send(result);
            }
        }
    }
}

/// Embed each text into a pooled, L2-normalized vector. A fresh embedding-mode context is created
/// per text (mean pooling) so sequences never share KV state; the model stays loaded across calls.
fn run_embed(
    backend: &LlamaBackend,
    model: &LlamaModel,
    params: &ModelParams,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, BackendError> {
    let n_ctx = if params.n_ctx > 0 {
        params.n_ctx
    } else {
        model.n_ctx_train()
    }
    .max(8);

    let mut out = Vec::with_capacity(texts.len());
    for text in texts {
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        if let Some(threads) = params.n_threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| BackendError::out_of_memory(format!("create embedding context: {e}")))?;

        let mut tokens = model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| BackendError::transient(format!("tokenize: {e}")))?;
        // Over-long inputs are clipped rather than failing the whole batch (embeddings tolerate it).
        if tokens.len() as u32 > n_ctx {
            tokens.truncate(n_ctx as usize);
        }

        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        for (i, token) in tokens.iter().enumerate() {
            // All tokens contribute to the pooled embedding, so each is marked an output.
            batch
                .add(*token, i as i32, &[0], true)
                .map_err(|e| BackendError::transient(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch).map_err(classify_decode)?;

        let emb = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| BackendError::transient(format!("read embeddings: {e}")))?;
        let mut v = emb.to_vec();
        l2_normalize(&mut v);
        out.push(v);
    }
    Ok(out)
}

/// L2-normalize in place (a no-op for the zero vector). Keeps cosine similarity a plain dot product.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Run one generation against the persistent [`GenSession`]: render the prompt, reuse the cached KV
/// prefix (re-decoding only the divergent suffix), then sample/decode token-by-token, streaming text.
///
/// On a cancellation or decode fault the KV cache may hold a partial/inconsistent sequence, so the
/// session's cache is reset (`clear_kv_cache` + `cached_tokens.clear()`) before returning — the next
/// call then prefills cleanly from scratch. A context-overflow leaves the prior cache intact (the
/// daemon compacts and retries with a shorter prompt that can still reuse the stable prefix).
fn run_generation<'a>(
    backend: &LlamaBackend,
    model: &'a LlamaModel,
    params: &ModelParams,
    req: &GenerateRequest,
    chunks: &UnboundedSender<BackendChunk>,
    cancel: &CancellationToken,
    session: &mut Option<GenSession<'a>>,
) -> Result<Usage, BackendError> {
    if cancel.is_cancelled() {
        return Err(BackendError::cancelled());
    }

    let prompt = render_prompt(model, req);

    let n_ctx = if params.n_ctx > 0 {
        params.n_ctx
    } else {
        model.n_ctx_train()
    }
    .max(8);

    // Drop the session when the context parameters changed: the KV layout depends on them, so a
    // cached prefix from a differently-shaped context is not reusable.
    if let Some(existing) = session.as_ref() {
        if existing.n_ctx != n_ctx
            || existing.n_threads != params.n_threads
            || existing.flash_attn != params.flash_attn
        {
            *session = None;
        }
    }

    // (Re)create the persistent context on first use or after an invalidation.
    if session.is_none() {
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(n_ctx));
        if let Some(threads) = params.n_threads {
            ctx_params = ctx_params.with_n_threads(threads as i32);
        }
        if params.flash_attn {
            ctx_params = ctx_params.with_flash_attention(true);
        }
        // A failed context allocation is almost always KV-cache VRAM/host OOM.
        let ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| BackendError::out_of_memory(format!("create context: {e}")))?;
        *session = Some(GenSession {
            ctx,
            cached_tokens: Vec::new(),
            n_ctx,
            n_threads: params.n_threads,
            flash_attn: params.flash_attn,
        });
    }
    let session = session.as_mut().expect("session present after (re)build");

    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| BackendError::transient(format!("tokenize: {e}")))?;
    let prompt_len = tokens.len() as u32;
    if prompt_len >= n_ctx {
        // The stable prefix is still valid for the compact-and-retry, so keep the cache as-is.
        return Err(BackendError::context_overflow(format!(
            "prompt {prompt_len} tokens >= context {n_ctx}"
        )));
    }

    // Prompt caching: keep the KV for the longest prefix shared with the previously decoded sequence
    // and re-decode only the suffix. Always re-decode at least the final token so the sampler has
    // fresh logits to work from (an identical prompt would otherwise leave nothing to decode).
    let mut common = find_common_prefix(&session.cached_tokens, &tokens);
    if common >= tokens.len() {
        common = tokens.len() - 1;
    }
    let reused = common as u64;

    // Evict the divergent tail (and any tokens generated last turn beyond the common prefix).
    if let Err(e) = session.ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None) {
        session.ctx.clear_kv_cache();
        session.cached_tokens.clear();
        return Err(BackendError::transient(format!("kv cache trim: {e}")));
    }

    // Prefill the divergent suffix, positions continuing from the reused prefix length.
    let suffix = &tokens[common..];
    let mut batch = LlamaBatch::new(suffix.len().max(1), 1);
    let last = suffix.len().saturating_sub(1);
    for (i, token) in suffix.iter().enumerate() {
        batch
            .add(*token, (common + i) as i32, &[0], i == last)
            .map_err(|e| BackendError::transient(format!("batch add: {e}")))?;
    }
    if let Err(e) = session.ctx.decode(&mut batch) {
        let failure = classify_decode(e);
        session.ctx.clear_kv_cache();
        session.cached_tokens.clear();
        return Err(failure);
    }

    let mut sampler = build_sampler(model, &req.sampling, req.constraint.as_ref());

    let budget = if req.max_tokens > 0 {
        req.max_tokens
    } else {
        DEFAULT_MAX_TOKENS
    }
    .min(n_ctx - prompt_len);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // The next free KV position after the prefilled prompt (prefix + suffix == full prompt).
    let mut n_cur = prompt_len as i32;
    let mut produced: u32 = 0;
    // Track the tokens generated this turn so the session's cached sequence matches the KV after the
    // call (prompt + generated), maximizing the next turn's reusable prefix.
    let mut generated: Vec<LlamaToken> = Vec::new();

    // With tools offered, buffer the full output so tool-call markup can be parsed out before any
    // text is surfaced; without tools, stream pieces live for low latency.
    let has_tools = !req.tools.is_empty();
    let mut buffered = String::new();

    while produced < budget {
        if cancel.is_cancelled() {
            session.ctx.clear_kv_cache();
            session.cached_tokens.clear();
            return Err(BackendError::cancelled());
        }

        let token = sampler.sample(&session.ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }

        let bytes = model
            .token_to_bytes(token, Special::Plaintext)
            .map_err(|e| BackendError::transient(format!("detokenize: {e}")))?;
        let mut piece = String::new();
        let _ = decoder.decode_to_string(&bytes, &mut piece, false);
        if !piece.is_empty() {
            if has_tools {
                buffered.push_str(&piece);
            } else if chunks.send(BackendChunk::Text(piece)).is_err() {
                // The consumer dropped the receiver (cancel/abort upstream); stop quietly.
                break;
            }
        }

        generated.push(token);
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| BackendError::transient(format!("batch add: {e}")))?;
        n_cur += 1;
        produced += 1;
        if let Err(e) = session.ctx.decode(&mut batch) {
            let failure = classify_decode(e);
            session.ctx.clear_kv_cache();
            session.cached_tokens.clear();
            return Err(failure);
        }
    }

    // Record the exact sequence now resident in the KV cache (prompt + generated tokens) so the next
    // request prefix-matches against it.
    session.cached_tokens = tokens;
    session.cached_tokens.extend(generated);

    if has_tools {
        let (cleaned, calls) = tooling::extract_tool_calls(&buffered);
        if !cleaned.is_empty() {
            let _ = chunks.send(BackendChunk::Text(cleaned));
        }
        for call in calls {
            if chunks.send(BackendChunk::Tool(call)).is_err() {
                break;
            }
        }
    }

    Ok(Usage {
        input_tokens: prompt_len as u64,
        output_tokens: produced as u64,
        cache_read_tokens: reused,
    })
}

/// Render the conversation into a prompt via the model's built-in chat template, falling back to a
/// simple role-tagged concatenation when the GGUF carries no template.
fn render_prompt(model: &LlamaModel, req: &GenerateRequest) -> String {
    let mut messages = Vec::new();
    let system = compose_system(req);
    if !system.is_empty() {
        if let Ok(m) = LlamaChatMessage::new("system".to_string(), system) {
            messages.push(m);
        }
    }
    for msg in &req.messages {
        if let Ok(m) =
            LlamaChatMessage::new(normalize_role(&msg.role).to_string(), msg.content.clone())
        {
            messages.push(m);
        }
    }
    let template: Option<String> = None;
    match model.apply_chat_template(template.as_deref(), &messages, true) {
        Ok(prompt) => prompt,
        Err(_) => fallback_prompt(req),
    }
}

/// The effective system prompt: the request's system text, prefixed with the tool-advertisement
/// preamble when tools are offered.
fn compose_system(req: &GenerateRequest) -> String {
    match tooling::tool_preamble(&req.tools) {
        Some(preamble) if req.system.is_empty() => preamble,
        Some(preamble) => format!("{preamble}\n{}", req.system),
        None => req.system.clone(),
    }
}

/// Map our roles onto the set chat templates accept; unknown roles degrade to `user`.
fn normalize_role(role: &str) -> &str {
    match role {
        "system" | "user" | "assistant" | "tool" => role,
        _ => "user",
    }
}

/// A template-free prompt: role-tagged lines plus a trailing assistant cue.
fn fallback_prompt(req: &GenerateRequest) -> String {
    let mut s = String::new();
    let system = compose_system(req);
    if !system.is_empty() {
        s.push_str(&system);
        s.push_str("\n\n");
    }
    for msg in &req.messages {
        s.push_str(normalize_role(&msg.role));
        s.push_str(": ");
        s.push_str(&msg.content);
        s.push('\n');
    }
    s.push_str("assistant: ");
    s
}

/// Build a sampler chain from the request's sampling params (greedy when temperature <= 0),
/// prepending a GBNF grammar sampler when the request carries a [`Constraint::Gbnf`]. The grammar
/// sampler is applied first so it masks invalid tokens before the selection samplers run. A
/// [`Constraint::Lark`] is for the mistral.rs engine — ignored here (with a warning).
fn build_sampler(model: &LlamaModel, s: &Sampling, constraint: Option<&Constraint>) -> LlamaSampler {
    let grammar = match constraint.map(|c| &c.gbnf) {
        Some(Some(g)) => Some(LlamaSampler::grammar(model, g, "root")),
        Some(None) => {
            tracing::warn!("llama: constraint carries no GBNF grammar; ignoring");
            None
        }
        None => None,
    };

    if s.temperature <= 0.0 {
        let mut chain = Vec::new();
        chain.extend(grammar);
        chain.push(LlamaSampler::greedy());
        return LlamaSampler::chain_simple(chain);
    }
    let mut chain = Vec::new();
    chain.extend(grammar);
    if s.top_k > 0 {
        chain.push(LlamaSampler::top_k(s.top_k as i32));
    }
    if s.top_p < 1.0 {
        chain.push(LlamaSampler::top_p(s.top_p, 1));
    }
    chain.push(LlamaSampler::temp(s.temperature));
    chain.push(LlamaSampler::dist(s.seed as u32));
    LlamaSampler::chain_simple(chain)
}

/// Classify a decode failure: a full KV cache / context is a context overflow, an allocator failure
/// is OOM, anything else is transient (worth one retry on a fresh worker).
fn classify_decode(e: DecodeError) -> BackendError {
    let msg = e.to_string();
    let lower = msg.to_ascii_lowercase();
    if lower.contains("kv") || lower.contains("n_ctx") || lower.contains("context") {
        BackendError::context_overflow(format!("decode: {msg}"))
    } else if lower.contains("memory") || lower.contains("alloc") || lower.contains("oom") {
        BackendError::out_of_memory(format!("decode: {msg}"))
    } else {
        BackendError::transient(format!("decode: {msg}"))
    }
}

#[cfg(test)]
mod tests {
    use super::find_common_prefix;

    #[test]
    fn common_prefix_counts_shared_leading_run() {
        // A growing conversation: the new prompt extends the previously cached sequence, so the whole
        // old sequence is the reusable prefix.
        assert_eq!(find_common_prefix(&[1, 2, 3], &[1, 2, 3, 4, 5]), 3);
        // A divergence mid-sequence stops the prefix at the first mismatch.
        assert_eq!(find_common_prefix(&[1, 2, 9, 4], &[1, 2, 3, 4]), 2);
        // No shared prefix (e.g. a different system prompt) reuses nothing.
        assert_eq!(find_common_prefix(&[9, 9], &[1, 2, 3]), 0);
        // An empty cache (first turn) reuses nothing.
        assert_eq!(find_common_prefix::<i32>(&[], &[1, 2, 3]), 0);
        // Identical sequences share their full length (the caller backs off by one before decoding).
        assert_eq!(find_common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
    }
}
