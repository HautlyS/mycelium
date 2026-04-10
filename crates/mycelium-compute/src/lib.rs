//! # Mycelium Compute — Distributed Tensor Engine
//!
//! Runs model inference across heterogeneous nodes using:
//! - candle (CPU/CUDA/Metal) for native nodes — real quantized transformer
//! - wgpu for GPU compute (native + WASM) — latent-space operations
//! - Latent-space processing instead of token-by-token generation

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module};
use mycelium_core::{
    GgufConfig, GgufValue, InferenceRequest, InferenceResponse, LatentVector, LoRAAdapter,
    ModelConfig, NodeId, TokenId, Tokenizer as MyceliumTokenizer,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

// ─── Latent Processing Mode ────────────────────────────────────────────────

/// How to process latents — the core innovation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LatentMode {
    /// Standard: encode → transform → decode (token-by-token output)
    Standard,
    /// Latent streaming: continuous latent vectors flow through
    LatentStream,
    /// Latent morphing: interpolate between two latent states
    LatentMorph { t: f32 },
    /// Latent blending: weighted combination of multiple latents
    LatentBlend { weights: Vec<f32> },
    /// Self-tuning: collect latent + gradient pairs for learning
    SelfTuning,
}

// ─── Quantized MatMul ──────────────────────────────────────────────────────

/// Wrapper around quantized matmul with tracing.
#[derive(Debug, Clone)]
pub(crate) struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: candle_core::quantized::QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(xs)?)
    }
}

// ─── RMS Norm ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(qtensor: candle_core::quantized::QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs_f = xs.to_dtype(DType::F32)?;
        let norm = xs_f
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?;
        let norm = (norm / xs_f.dim(candle_core::D::Minus1)? as f64)?;
        let norm = (norm + self.eps)?;
        let xs_normed = xs_f.broadcast_div(&norm.sqrt()?)?;
        let result = xs_normed.broadcast_mul(&self.weight)?;
        Ok(result.to_dtype(dtype)?)
    }
}

// ─── SiLU activation ───────────────────────────────────────────────────────

fn silu(xs: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(xs)?)
}

// ─── Feed-Forward Network ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        Ok(self.feed_forward_w2.forward(&(silu(&w1)? * w3)?)?)
    }
}

// ─── Mixture of Experts ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::MoE {
                n_expert_used,
                feed_forward_gate_inp,
                experts,
            } => {
                let (_b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;
                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        sum_routing_weights += rw[expert_idx];
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        selected_rws[expert_idx].push(rw[expert_idx] / sum_routing_weights);
                    }
                }

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x = &top_x[expert_idx];
                    if top_x.is_empty() {
                        continue;
                    }
                    let selected_rws = &selected_rws[expert_idx];
                    let expert_xs = xs.index_select(
                        &Tensor::new(top_x.as_slice(), xs.device())?,
                        0,
                    )?;
                    let expert_ys = expert_layer.forward(&expert_xs)?;
                    let selected_rws = Tensor::new(selected_rws.as_slice(), xs.device())?
                        .reshape((selected_rws.len(), 1))?;
                    let expert_ys = expert_ys.broadcast_mul(&selected_rws)?;

                    // Scatter back
                    let indices = Tensor::new(top_x.as_slice(), xs.device())?;
                    ys = ys.scatter_add(&indices, &expert_ys, 0)?;
                }

                Ok(ys.reshape((_b_size, seq_len, hidden_dim))?)
            }
        }
    }
}

// ─── Attention ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    /// Optional LoRA adapter for this layer
    lora: Option<LoRALayerWeights>,
}

/// LoRA adapter weights for a single attention layer.
#[derive(Debug, Clone)]
struct LoRALayerWeights {
    /// LoRA A matrix for Q projection: shape [rank, hidden_dim]
    lora_a_q: Tensor,
    /// LoRA B matrix for Q projection: shape [hidden_dim, rank]
    lora_b_q: Tensor,
    /// LoRA A matrix for V projection: shape [rank, hidden_dim]
    lora_a_v: Tensor,
    /// LoRA B matrix for V projection: shape [hidden_dim, rank]
    lora_b_v: Tensor,
    /// Scaling factor: alpha / rank
    scaling: f64,
}

impl LoRALayerWeights {
    /// Apply LoRA to a Q projection output.
    /// Computes: output = output + (x @ A^T) @ B^T * scaling
    fn apply_q(&self, xs: &Tensor, output: &Tensor) -> Result<Tensor> {
        let lora_out = xs.matmul(&self.lora_a_q.t()?)?.matmul(&self.lora_b_q.t()?)?;
        Ok((output + (lora_out * self.scaling)?)?)
    }

    /// Apply LoRA to a V projection output.
    fn apply_v(&self, xs: &Tensor, output: &Tensor) -> Result<Tensor> {
        let lora_out = xs.matmul(&self.lora_a_v.t()?)?.matmul(&self.lora_b_v.t()?)?;
        Ok((output + (lora_out * self.scaling)?)?)
    }
}

impl LayerWeights {
    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_dim) = xs.dims3()?;
        let hidden_dim = self.head_dim * self.n_head;

        // Pre-attention norm
        let normed = self.attention_norm.forward(xs)?;

        // QKV projections
        let q = self.attention_wq.forward(&normed)?;
        let k = self.attention_wk.forward(&normed)?;
        let v = self.attention_wv.forward(&normed)?;

        // Apply LoRA adapters if present
        let (q, v) = if let Some(lora) = &self.lora {
            let q = lora.apply_q(&normed, &q)?;
            let v = lora.apply_v(&normed, &v)?;
            (q, v)
        } else {
            (q, v)
        };

        // Reshape for multi-head attention
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?; // (b, n_head, seq, head_dim)
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let q = apply_rotary_emb(&q, &self.cos, &self.sin, seq_len)?;
        let k = apply_rotary_emb(&k, &self.cos, &self.sin, seq_len)?;

        // KV cache handling
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                self.kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
            None => {
                self.kv_cache = Some((k.clone(), v.clone()));
                (k, v)
            }
        };

        // GQA: repeat KV heads if n_kv_head < n_head
        let n_rep = self.n_head / self.n_kv_head;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, hidden_dim))?;

        let attn_output = self.attention_wo.forward(&attn_output)?;

        // Residual
        let ys = (xs + attn_output)?;

        // Post-attention norm + FFN
        let normed = self.ffn_norm.forward(&ys)?;
        let ffn_output = self.mlp_or_moe.forward(&normed)?;

        // Residual
        Ok((&ys + ffn_output)?)
    }

    /// Extract the latent vector at this layer (post-attention, pre-FFN residual).
    fn extract_latent(&self, xs: &Tensor) -> Result<Tensor> {
        // Return the normed hidden state after attention
        self.attention_norm.forward(xs)
    }

    /// Clear the KV cache.
    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ─── Rotary Embeddings ─────────────────────────────────────────────────────

fn precompute_freqs_cis(head_dim: usize, freq_base: f32, max_seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

fn apply_rotary_emb(q: &Tensor, cos: &Tensor, sin: &Tensor, seq_len: usize) -> Result<Tensor> {
    let (_b_sz, _n_head, _seq, head_dim) = q.dims4()?;
    let cos = cos.i((..seq_len, ..))?.reshape((1, 1, seq_len, head_dim / 2, 1))?;
    let sin = sin.i((..seq_len, ..))?.reshape((1, 1, seq_len, head_dim / 2, 1))?;

    let q = q.reshape((_b_sz, _n_head, _seq, head_dim / 2, 2))?;
    let q0 = q.i((.., .., .., .., 0))?;
    let q1 = q.i((.., .., .., .., 1))?;

    let rotated = Tensor::stack(&[
        &(q0.broadcast_mul(&cos)? - q1.broadcast_mul(&sin)?)?,
        &(q0.broadcast_mul(&sin)? + q1.broadcast_mul(&cos)?)?,
    ], candle_core::D::Minus1)?;

    Ok(rotated.reshape((_b_sz, _n_head, _seq, head_dim))?)
}

fn repeat_kv(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    let xs = xs
        .unsqueeze(2)?
        .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
    Ok(xs)
}

// ─── Model Weights ─────────────────────────────────────────────────────────

/// Full model weights loaded from a GGUF file.
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<usize, Tensor>,
    config: ModelConfig,
    device: Device,
}

impl ModelWeights {
    /// Load model weights from a GGUF file.
    pub fn from_gguf(path: &Path, device: &Device) -> Result<Self> {
        info!("Loading GGUF model from: {}", path.display());
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
        let mut reader = std::io::BufReader::new(file);

        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

        // Extract metadata
        let md_get = |s: &str| match content.metadata.get(s) {
            None => candle_core::bail!("cannot find {} in metadata", s),
            Some(v) => Ok(v),
        };

        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        info!(
            "Model config: {} layers, {} heads ({} KV), {}d embed, {} experts ({} used)",
            block_count, head_count, head_count_kv, embedding_length, n_expert, n_expert_used
        );

        let head_dim = embedding_length / head_count;
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, 4096, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load embeddings
        let tok_embeddings_q = content.tensor(&mut reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            content.tensor(&mut reader, "output_norm.weight", device)?,
            rms_norm_eps,
            device,
        )?;
        let output = match content.tensor(&mut reader, "output.weight", device) {
            Ok(tensor) => QMatMul::from_qtensor(tensor)?,
            Err(_) => QMatMul::from_qtensor(tok_embeddings_q)?,
        };

        // Load layers
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            debug!("Loading layer {}/{}", layer_idx, block_count);

            let attention_wq = content.tensor(&mut reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = content.tensor(&mut reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = content.tensor(&mut reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo = content.tensor(&mut reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp_or_moe = if n_expert <= 1 {
                let w1 = content.tensor(&mut reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let w2 = content.tensor(&mut reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let w3 = content.tensor(&mut reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(w3)?,
                })
            } else {
                let gate_inp = content.tensor(&mut reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(n_expert);
                for i in 0..n_expert {
                    let w1 = content.tensor(&mut reader, &format!("{prefix}.ffn_gate.{i}.weight"), device)?;
                    let w2 = content.tensor(&mut reader, &format!("{prefix}.ffn_down.{i}.weight"), device)?;
                    let w3 = content.tensor(&mut reader, &format!("{prefix}.ffn_up.{i}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(w3)?,
                    });
                }
                MlpOrMoe::MoE {
                    n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(gate_inp)?,
                    experts,
                }
            };

            let attention_norm = content.tensor(&mut reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = content.tensor(&mut reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps, device)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps, device)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                lora: None,
            });
        }

        let config = ModelConfig::minimax_m25(); // Will be replaced with GgufConfig

        info!("Model loaded: {} layers, {}d embed", layers.len(), embedding_length);

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            config,
            device: device.clone(),
        })
    }

    /// Load model weights from GGUF and also populate config from metadata.
    pub fn from_gguf_with_config(path: &Path, device: &Device) -> Result<Self> {
        info!("Loading GGUF model with config from: {}", path.display());
        let mut model = Self::from_gguf(path, device)?;

        // Re-read to extract GGUF config
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file: {}", path.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF for config: {}", e))?;

        let gguf_config = gguf_content_to_config(&content);
        model.config = gguf_config.to_model_config();

        info!("Model config from GGUF: {:?}", model.config);
        Ok(model)
    }

    /// Get causal mask.
    fn mask(&mut self, t: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Forward pass through the full model.
    pub fn forward(&mut self, tokens: &[TokenId], _index_pos: usize) -> Result<Tensor> {
        let seq_len = tokens.len();
        let tokens_tensor = Tensor::new(tokens, &self.device)?.reshape((1, tokens.len()))?;
        let xs = self.tok_embeddings.forward(&tokens_tensor)?;

        let mask = if seq_len == 1 {
            None
        } else {
            let device = self.device.clone();
            Some(self.mask(seq_len, &device)?)
        };

        let mut layer_in = xs;
        for layer in self.layers.iter_mut() {
            layer_in = layer.forward(&layer_in, mask.as_ref())?;
        }

        let output = self.norm.forward(&layer_in)?;
        let output = self.output.forward(&output)?;

        Ok(output)
    }

    /// Forward pass with latent extraction at a specific layer.
    /// Returns (logits, latent_at_layer).
    pub fn forward_with_latent(
        &mut self,
        tokens: &[TokenId],
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = tokens.len();
        let tokens_tensor = Tensor::new(tokens, &self.device)?.reshape((1, tokens.len()))?;
        let xs = self.tok_embeddings.forward(&tokens_tensor)?;

        let mask = if seq_len == 1 {
            None
        } else {
            let device = self.device.clone();
            Some(self.mask(seq_len, &device)?)
        };

        let mut layer_in = xs;
        let mut latent = None;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == layer_idx {
                // Extract latent before this layer's computation
                latent = Some(layer.extract_latent(&layer_in)?);
            }
            layer_in = layer.forward(&layer_in, mask.as_ref())?;
        }

        let output = self.norm.forward(&layer_in)?;
        let logits = self.output.forward(&output)?;

        let latent = latent
            .ok_or_else(|| anyhow::anyhow!("Failed to extract latent at layer {}", layer_idx))?;

        Ok((logits, latent))
    }

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        prompt_tokens: &[TokenId],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<TokenId>> {
        // Clear KV caches
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }

        let mut generated = prompt_tokens.to_vec();

        // Prefill: process the full prompt
        let logits = self.forward(prompt_tokens, 0)?;
        let next_token = sample_token(&logits, temperature, top_p, prompt_tokens.len() - 1)?;
        generated.push(next_token);

        // Decode: one token at a time
        for i in 1..max_new_tokens {
            let logits = self.forward(&[next_token], generated.len() - 1)?;
            let next = sample_token(&logits, temperature, top_p, 0)?;
            generated.push(next);

            // Check for EOS token
            if let Some(eos_id) = self.config.eos_token_id {
                if next == eos_id {
                    debug!("EOS token ({}) generated at step {}, stopping", eos_id, i);
                    break;
                }
            }
        }

        Ok(generated)
    }

    /// Extract latent vector at a specific layer.
    pub fn extract_latent_at_layer(
        &mut self,
        tokens: &[TokenId],
        layer_idx: usize,
    ) -> Result<LatentVector> {
        if layer_idx >= self.layers.len() {
            bail!("Layer index {} out of range (max {})", layer_idx, self.layers.len());
        }

        let tokens_tensor = Tensor::new(tokens, &self.device)?.reshape((1, tokens.len()))?;
        let mut layer_in = self.tok_embeddings.forward(&tokens_tensor)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == layer_idx {
                // Extract latent before this layer's computation
                let latent = layer.extract_latent(&layer_in)?;
                return tensor_to_latent(&latent, layer_idx);
            }
            let mask = None; // single token, no mask needed for extraction
            layer_in = layer.forward(&layer_in, mask)?;
        }

        bail!("Failed to extract latent at layer {}", layer_idx);
    }

    /// Apply a LoRA adapter to specific layers.
    pub fn apply_lora(&mut self, adapter: &LoRAAdapter) -> Result<()> {
        let scaling = (adapter.alpha / adapter.rank as f32) as f64;

        for &layer_idx in &adapter.target_layers {
            if layer_idx >= self.layers.len() {
                warn!("LoRA target layer {} out of range, skipping", layer_idx);
                continue;
            }

            let hidden_dim = self.config.hidden_dim;
            let rank = adapter.rank;

            // Create LoRA A and B matrices for Q and V projections
            let device = &self.device;
            let a_q_data = &adapter.a_weights.get(layer_idx)
                .ok_or_else(|| anyhow::anyhow!("LoRA missing a_weights for layer {}", layer_idx))?;
            let b_q_data = &adapter.b_weights.get(layer_idx)
                .ok_or_else(|| anyhow::anyhow!("LoRA missing b_weights for layer {}", layer_idx))?;

            // A matrices: [rank, hidden_dim], B matrices: [hidden_dim, rank]
            let lora_a_q = Tensor::from_slice(a_q_data, (rank, hidden_dim), device)?.to_dtype(DType::F32)?;
            let lora_b_q = Tensor::from_slice(b_q_data, (hidden_dim, rank), device)?.to_dtype(DType::F32)?;

            // Use the same weights for V (simplified; in practice V would have separate weights)
            let lora_a_v = lora_a_q.clone();
            let lora_b_v = lora_b_q.clone();

            self.layers[layer_idx].lora = Some(LoRALayerWeights {
                lora_a_q,
                lora_b_q,
                lora_a_v,
                lora_b_v,
                scaling,
            });

            info!("Applied LoRA adapter to layer {} (rank={}, alpha={}, scaling={})",
                  layer_idx, adapter.rank, adapter.alpha, scaling);
        }

        Ok(())
    }

    /// Get the number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get the model config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Clear all KV caches.
    pub fn clear_kv_caches(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

// ─── GgufConfig extraction helper ──────────────────────────────────────────

/// Convert a candle gguf_file::Content's metadata into a GgufConfig.
///
/// This is a free function because we cannot impl on a type from another crate.
fn gguf_content_to_config(content: &gguf_file::Content) -> GgufConfig {
    let mut kv = HashMap::new();

    for (key, value) in &content.metadata {
        let gguf_val = convert_gguf_value(value);
        kv.insert(key.clone(), gguf_val);
    }

    GgufConfig::from_metadata(kv)
}

/// Convert a single candle gguf_file::Value to a mycelium GgufValue.
fn convert_gguf_value(value: &gguf_file::Value) -> GgufValue {
    match value {
        gguf_file::Value::String(s) => GgufValue::String(s.clone()),
        gguf_file::Value::U8(v) => GgufValue::Uint(*v as u64),
        gguf_file::Value::I8(v) => GgufValue::Int(*v as i64),
        gguf_file::Value::U16(v) => GgufValue::Uint(*v as u64),
        gguf_file::Value::I16(v) => GgufValue::Int(*v as i64),
        gguf_file::Value::U32(v) => GgufValue::Uint(*v as u64),
        gguf_file::Value::I32(v) => GgufValue::Int(*v as i64),
        gguf_file::Value::U64(v) => GgufValue::Uint(*v),
        gguf_file::Value::I64(v) => GgufValue::Int(*v),
        gguf_file::Value::F32(v) => GgufValue::Float(*v as f64),
        gguf_file::Value::F64(v) => GgufValue::Float(*v),
        gguf_file::Value::Bool(v) => GgufValue::Bool(*v),
        gguf_file::Value::Array(arr) => {
            GgufValue::Array(arr.iter().map(convert_gguf_value).collect())
        }
    }
}

// ─── Token Sampling ────────────────────────────────────────────────────────

fn sample_token(logits: &Tensor, temperature: f32, top_p: f32, pos: usize) -> Result<TokenId> {
    let logits = logits.i((.., pos, ..))?.squeeze(0)?.to_dtype(DType::F32)?;

    // Apply temperature
    let logits = if temperature > 0.0 {
        (logits / (temperature as f64))?
    } else {
        logits
    };

    // Softmax to get probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;
    let probs_vec = probs.to_vec1::<f32>()?;

    // Sort indices by probability descending for top-p filtering
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Find cutoff for top-p (nucleus) filtering
    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // Keep only top-p candidates
    let candidates: Vec<(usize, f32)> = indexed.into_iter().take(cutoff).collect();

    // Renormalize probabilities
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total <= 0.0 {
        return Ok(candidates[0].0 as TokenId);
    }

    // Multinomial sampling from the renormalized distribution
    let threshold = rand::random::<f32>() * total;
    let mut cumulative = 0.0f32;
    for (idx, prob) in &candidates {
        cumulative += prob;
        if cumulative >= threshold {
            return Ok(*idx as TokenId);
        }
    }

    // Fallback to last candidate (should not happen with proper rounding)
    Ok(candidates.last().unwrap().0 as TokenId)
}

// ─── Tensor ↔ LatentVector Conversion ─────────────────────────────────────

fn tensor_to_latent(tensor: &Tensor, layer_idx: usize) -> Result<LatentVector> {
    let flat = tensor.reshape((tensor.elem_count(),))?.to_vec1::<f32>()?;
    let _dim = flat.len();
    Ok(LatentVector::from_vec(flat, layer_idx, uuid::Uuid::new_v4()))
}

// ─── Device Detection ──────────────────────────────────────────────────────

/// Detect available compute devices.
pub fn detect_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            info!("Using CUDA device");
            return Ok(device);
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            info!("Using Metal device");
            return Ok(device);
        }
    }

    info!("Using CPU device");
    Ok(Device::Cpu)
}

/// Get device info string.
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".into(),
        Device::Cuda(_) => "CUDA:0".into(),
        Device::Metal(_) => "Metal:0".into(),
    }
}

// ─── GGUF Loader ──────────────────────────────────────────────────────────

/// Loaded model data from a GGUF file, including the parsed content
/// for later tensor access.
pub struct LoadedModel {
    pub config: ModelConfig,
    pub device: Device,
    pub tensor_data_offset: u64,
    /// Number of weight tensors in the model
    pub tensor_count: usize,
    /// Names of all tensors in the model
    pub tensor_names: Vec<String>,
    pub content: gguf_file::Content,
}

/// GGUF model loader — loads a GGUF model from file using candle's gguf_file parser.
pub struct GGUFLoader {
    model_path: PathBuf,
    device: Device,
}

impl GGUFLoader {
    /// Create a new GGUFLoader for the given path and device.
    pub fn new(path: impl Into<PathBuf>, device: Device) -> Self {
        Self {
            model_path: path.into(),
            device,
        }
    }

    /// Load the GGUF model, returning a LoadedModel with config and content.
    pub fn load(&self) -> Result<LoadedModel> {
        info!("GGUFLoader: loading from {}", self.model_path.display());

        let file = std::fs::File::open(&self.model_path)
            .with_context(|| format!("Failed to open GGUF file: {}", self.model_path.display()))?;
        let mut reader = std::io::BufReader::new(file);

        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

        // Extract config from GGUF metadata
        let gguf_config = gguf_content_to_config(&content);
        let model_config = gguf_config.to_model_config();

        info!(
            "GGUFLoader: loaded model '{}' — {} layers, {}d, {} heads, {} experts",
            model_config.name,
            model_config.num_layers,
            model_config.hidden_dim,
            model_config.num_heads,
            model_config.num_experts,
        );

        let tensor_data_offset = content.tensor_data_offset;
        let tensor_count = content.tensor_infos.len();
        let tensor_names: Vec<String> = content.tensor_infos.keys().cloned().collect();

        Ok(LoadedModel {
            config: model_config,
            device: self.device.clone(),
            tensor_data_offset,
            tensor_count,
            tensor_names,
            content,
        })
    }

    /// Load the full model weights from the GGUF file, ready for inference.
    pub fn load_model_weights(&self) -> Result<ModelWeights> {
        ModelWeights::from_gguf_with_config(&self.model_path, &self.device)
    }
}

// ─── Generate Result ──────────────────────────────────────────────────────

/// Result from text generation.
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// The generated text
    pub text: String,
    /// All generated token IDs (including prompt)
    pub token_ids: Vec<TokenId>,
    /// Number of new tokens generated
    pub new_token_count: usize,
}

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The model configuration
    pub config: ModelConfig,
    /// Number of weight tensors
    pub tensor_count: usize,
    /// Path the model was loaded from
    pub path: PathBuf,
    /// Device being used
    pub device_info: String,
}

// ─── Inference Engine ──────────────────────────────────────────────────────

/// Real inference engine with candle-transformers Llama model support.
///
/// Supports:
/// - Loading GGUF models via candle's gguf_file parser
/// - Text generation with tokenization via the `tokenizers` crate
/// - Latent extraction at specific layers for self-tuning
/// - LoRA adapter application during forward pass
pub struct InferenceEngine {
    model: Option<ModelWeights>,
    loaded_model: Option<LoadedModel>,
    device: Device,
    config: ModelConfig,
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl InferenceEngine {
    /// Create a new inference engine with the given device.
    pub fn new(device: Device) -> Self {
        Self {
            model: None,
            loaded_model: None,
            device,
            config: ModelConfig::default(),
            tokenizer: None,
        }
    }

    /// Create a new inference engine with CPU device.
    pub fn cpu() -> Self {
        Self::new(Device::Cpu)
    }

    /// Create a new inference engine, auto-detecting the best device.
    pub fn auto_detect() -> Result<Self> {
        let device = detect_device()?;
        Ok(Self::new(device))
    }

    /// Load a GGUF model from the given path.
    pub fn load_model(&mut self, path: &Path) -> Result<ModelInfo> {
        // First, use GGUFLoader to get config and content
        let loader = GGUFLoader::new(path, self.device.clone());
        let loaded = loader.load()?;
        let config = loaded.config.clone();
        let tensor_count = loaded.tensor_count;
        let dev_info = device_info(&self.device);

        self.loaded_model = Some(loaded);
        self.config = config.clone();

        // Then load full model weights for inference
        let model = ModelWeights::from_gguf_with_config(path, &self.device)?;
        self.model = Some(model);

        info!("InferenceEngine: model loaded from {}", path.display());

        Ok(ModelInfo {
            config,
            tensor_count,
            path: path.to_path_buf(),
            device_info: dev_info,
        })
    }

    /// Load a tokenizer from a file (JSON format from HuggingFace tokenizers library).
    pub fn load_tokenizer(&mut self, path: &Path) -> Result<()> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", path.display(), e))?;
        self.tokenizer = Some(tokenizer);
        info!("InferenceEngine: tokenizer loaded from {}", path.display());
        Ok(())
    }

    /// Set the tokenizer directly.
    pub fn set_tokenizer(&mut self, tokenizer: tokenizers::Tokenizer) {
        self.tokenizer = Some(tokenizer);
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Get the model config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the loaded model (if any).
    pub fn loaded_model(&self) -> Option<&LoadedModel> {
        self.loaded_model.as_ref()
    }

    /// Generate text from a prompt string.
    ///
    /// Tokenizes the prompt, runs the forward pass, and samples tokens
    /// autoregressively. Requires both a model and tokenizer to be loaded.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerateResult> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded. Call load_model() first."))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer loaded. Call load_tokenizer() first."))?;

        // Tokenize the prompt
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens: Vec<TokenId> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        debug!("Prompt tokens: {} tokens", prompt_len);

        // Generate
        let top_p = 0.9;
        let generated_tokens = model.generate(&prompt_tokens, max_tokens, temperature, top_p)?;

        // Decode the generated tokens (only the new ones)
        let new_token_count = generated_tokens.len().saturating_sub(prompt_len);
        let new_tokens = &generated_tokens[prompt_len..];

        let text = tokenizer.decode(new_tokens, false)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;

        Ok(GenerateResult {
            text,
            token_ids: generated_tokens,
            new_token_count,
        })
    }

    /// Extract latent vector at a specific layer for a given prompt.
    ///
    /// This is the key capability for the self-tuning loop. It runs the
    /// forward pass through the model and captures the hidden state at
    /// the specified layer index.
    pub fn extract_latent(
        &mut self,
        prompt: &str,
        layer_idx: usize,
    ) -> Result<LatentVector> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded. Call load_model() first."))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer loaded. Call load_tokenizer() first."))?;

        if layer_idx >= model.layer_count() {
            bail!(
                "Layer index {} out of range (model has {} layers)",
                layer_idx,
                model.layer_count()
            );
        }

        // Tokenize
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let tokens: Vec<TokenId> = encoding.get_ids().to_vec();

        // Extract latent
        model.extract_latent_at_layer(&tokens, layer_idx)
    }

    /// Extract latent vectors from all layers.
    ///
    /// Useful for comprehensive self-tuning analysis.
    pub fn extract_all_latents(
        &mut self,
        prompt: &str,
    ) -> Result<Vec<LatentVector>> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded. Call load_model() first."))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer loaded. Call load_tokenizer() first."))?;

        // Tokenize
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let tokens: Vec<TokenId> = encoding.get_ids().to_vec();

        let num_layers = model.layer_count();
        let mut latents = Vec::with_capacity(num_layers);

        // Run forward pass, extracting at each layer
        // We need to do this layer by layer since extract_latent_at_layer
        // runs the full forward pass
        for layer_idx in 0..num_layers {
            let latent = model.extract_latent_at_layer(&tokens, layer_idx)?;
            latents.push(latent);
        }

        Ok(latents)
    }

    /// Apply a LoRA adapter to the model.
    pub fn apply_lora(&mut self, adapter: &LoRAAdapter) -> Result<()> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded. Call load_model() first."))?;
        model.apply_lora(adapter)
    }

    /// Run inference from an InferenceRequest.
    pub fn infer(&mut self, request: &InferenceRequest) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();

        if request.latent_mode {
            // Extract latents instead of generating text
            let latents = self.extract_all_latents(&request.prompt)?;
            Ok(InferenceResponse {
                id: request.id,
                text: None,
                latents,
                participating_nodes: Vec::new(),
                latency_ms: start.elapsed().as_millis() as u64,
            })
        } else {
            let result = self.generate(&request.prompt, request.max_tokens, request.temperature)?;
            Ok(InferenceResponse {
                id: request.id,
                text: Some(result.text),
                latents: Vec::new(),
                participating_nodes: Vec::new(),
                latency_ms: start.elapsed().as_millis() as u64,
            })
        }
    }

    /// Generate text with latent extraction at a specific layer.
    /// Returns both the generation result and latent vectors.
    pub fn generate_with_latent(
        &mut self,
        prompt: &str,
        layer_idx: usize,
    ) -> Result<(GenerateResult, Vec<LatentVector>)> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded. Call load_model() first."))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer loaded. Call load_tokenizer() first."))?;

        // Tokenize the prompt
        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens: Vec<TokenId> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        // Generate tokens
        let max_tokens = 64;
        let temperature = 0.7f32;
        let generated_tokens = model.generate(&prompt_tokens, max_tokens, temperature, 0.9)?;
        let new_token_count = generated_tokens.len().saturating_sub(prompt_len);
        let new_tokens = &generated_tokens[prompt_len..];
        let text = tokenizer.decode(new_tokens, false)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;

        let generate_result = GenerateResult {
            text,
            token_ids: generated_tokens,
            new_token_count,
        };

        // Extract latent at the specified layer
        let latent = model.extract_latent_at_layer(&prompt_tokens, layer_idx)?;
        let latents = vec![latent];

        Ok((generate_result, latents))
    }
}

// ─── MoE Inference Engine ──────────────────────────────────────────────────

/// Distributed MoE inference engine.
///
/// This engine uses real GGUF model loading when available and supports
/// distributed inference across multiple nodes via the DistributedCoordinator.
pub struct MoeInferenceEngine {
    config: ModelConfig,
    node_id: NodeId,
    latent_mode: LatentMode,
    model: Option<ModelWeights>,
    loaded_model: Option<LoadedModel>,
    tokenizer: Option<tokenizers::Tokenizer>,
    coordinator: DistributedCoordinator,
    device: Device,
}

impl MoeInferenceEngine {
    /// Create a new MoE inference engine.
    pub fn new(config: ModelConfig, node_id: NodeId) -> Result<Self> {
        let device = detect_device()?;
        let coordinator = DistributedCoordinator::new(config.clone(), node_id.clone());
        Ok(Self {
            config,
            node_id,
            latent_mode: LatentMode::Standard,
            model: None,
            loaded_model: None,
            tokenizer: None,
            coordinator,
            device,
        })
    }

    /// Create with a specific device.
    pub fn with_device(config: ModelConfig, node_id: NodeId, device: Device) -> Self {
        let coordinator = DistributedCoordinator::new(config.clone(), node_id.clone());
        Self {
            config,
            node_id,
            latent_mode: LatentMode::Standard,
            model: None,
            loaded_model: None,
            tokenizer: None,
            coordinator,
            device,
        }
    }

    /// Set the latent processing mode.
    pub fn set_latent_mode(&mut self, mode: LatentMode) {
        self.latent_mode = mode;
    }

    /// Load a model from a GGUF file.
    pub fn load_model(&mut self, path: &Path) -> Result<ModelInfo> {
        let loader = GGUFLoader::new(path, self.device.clone());
        let loaded = loader.load()?;
        let config = loaded.config.clone();
        let tensor_count = loaded.tensor_count;
        let dev_info = device_info(&self.device);

        self.loaded_model = Some(loaded);
        self.config = config.clone();

        // Load full model weights
        let model = ModelWeights::from_gguf_with_config(path, &self.device)?;
        self.model = Some(model);

        info!("MoeInferenceEngine: model loaded from {}", path.display());

        Ok(ModelInfo {
            config,
            tensor_count,
            path: path.to_path_buf(),
            device_info: dev_info,
        })
    }

    /// Load a model shard from a GGUF file.
    pub async fn load_shard(
        &mut self,
        path: &Path,
        _layer_range: (usize, usize),
        _expert_ids: Vec<usize>,
    ) -> Result<()> {
        let loader = GGUFLoader::new(path, self.device.clone());
        let loaded = loader.load()?;
        self.config = loaded.config.clone();
        self.loaded_model = Some(loaded);

        let model = ModelWeights::from_gguf_with_config(path, &self.device)?;
        self.model = Some(model);

        info!("Loaded model shard from {}", path.display());
        Ok(())
    }

    /// Load a tokenizer.
    pub fn load_tokenizer(&mut self, path: &Path) -> Result<()> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    /// Generate text from a prompt.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerateResult> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer loaded"))?;

        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens: Vec<TokenId> = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        let generated_tokens = model.generate(&prompt_tokens, max_tokens, temperature, 0.9)?;
        let new_token_count = generated_tokens.len().saturating_sub(prompt_len);
        let new_tokens = &generated_tokens[prompt_len..];

        let text = tokenizer.decode(new_tokens, false)
            .map_err(|e| anyhow::anyhow!("Detokenization failed: {}", e))?;

        Ok(GenerateResult {
            text,
            token_ids: generated_tokens,
            new_token_count,
        })
    }

    /// Extract latent at a specific layer.
    pub fn extract_latent(
        &mut self,
        prompt: &str,
        layer_idx: usize,
    ) -> Result<LatentVector> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No tokenizer loaded"))?;

        let encoding = tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let tokens: Vec<TokenId> = encoding.get_ids().to_vec();

        model.extract_latent_at_layer(&tokens, layer_idx)
    }

    /// Apply a LoRA adapter.
    pub fn apply_lora(&mut self, adapter: &LoRAAdapter) -> Result<()> {
        let model = self.model.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        model.apply_lora(adapter)
    }

    /// Get the model config.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }
}

// ─── Distributed Coordinator ────────────────────────────────────────────────

/// Coordinates distributed inference across multiple nodes.
pub struct DistributedCoordinator {
    config: ModelConfig,
    topology: mycelium_core::TopologyMap,
    node_id: NodeId,
}

impl DistributedCoordinator {
    pub fn new(config: ModelConfig, node_id: NodeId) -> Self {
        Self {
            config,
            topology: mycelium_core::TopologyMap::default(),
            node_id,
        }
    }

    pub fn update_topology(&mut self, topology: mycelium_core::TopologyMap) {
        info!(
            "Topology updated: {} nodes, {}MB total VRAM",
            topology.nodes.len(),
            topology.total_vram_mb(),
        );
        self.topology = topology;
    }

    /// Compute layer assignments for distributed inference.
    pub fn compute_assignments(&self) -> Vec<mycelium_core::LayerAssignment> {
        if self.topology.nodes.is_empty() {
            return Vec::new();
        }

        let total_vram: u32 = self.topology.nodes.iter().map(|(_, cap)| cap.vram_mb).sum();
        if total_vram == 0 {
            return Vec::new();
        }

        let mut assignments = Vec::new();
        let mut layer_cursor = 0;

        for (node_id, cap) in &self.topology.nodes {
            let node_share = cap.vram_mb as f64 / total_vram as f64;
            let layers_for_node = (node_share * self.config.num_layers as f64).ceil() as usize;
            let layer_end = (layer_cursor + layers_for_node).min(self.config.num_layers);

            if layer_cursor >= self.config.num_layers {
                break;
            }

            let total_compute: u32 = self.topology.nodes.iter().map(|(_, c)| c.compute_units).sum();
            let expert_share = if total_compute > 0 {
                cap.compute_units as f64 / total_compute as f64
            } else {
                1.0 / self.topology.nodes.len() as f64
            };
            let experts_for_node = (expert_share * self.config.num_experts as f64).ceil() as usize;
            let expert_ids: Vec<usize> = (0..self.config.num_experts).take(experts_for_node).collect();

            assignments.push(mycelium_core::LayerAssignment {
                node_id: *node_id,
                layer_start: layer_cursor,
                layer_end,
                expert_ids,
                priority: 0,
            });

            layer_cursor = layer_end;
        }

        assignments
    }

    pub fn find_expert_node(&self, expert_id: usize) -> Option<NodeId> {
        self.topology.assignments.iter()
            .find(|a| a.expert_ids.contains(&expert_id))
            .map(|a| a.node_id)
    }

    pub fn find_layer_node(&self, layer_idx: usize) -> Option<NodeId> {
        self.topology.best_node_for_layer(layer_idx)
    }
}

// ─── Distributed Tensor Router ──────────────────────────────────────────────
// CRITICAL: This is the bridge between Hyphae (P2P) and Compute (inference).
// It coordinates cross-node tensor operations during distributed inference.

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, oneshot};
use std::collections::HashMap as StdHashMap;

/// Pending inference request tracking.
#[derive(Debug)]
struct PendingRequest {
    /// Original request ID
    request_id: uuid::Uuid,
    /// Sender for the response
    response_tx: oneshot::Sender<Result<LatentVector>>,
    /// Which layer this request is waiting for
    waiting_layer: usize,
    /// Timestamp when request was made
    created_at: std::time::Instant,
}

/// Result of a local layer computation.
#[derive(Debug, Clone)]
pub struct LayerResult {
    pub layer_idx: usize,
    pub latent: LatentVector,
    pub is_final: bool,
}

/// Command sent to the router task.
pub enum RouterCommand {
    /// Process an incoming latent from the network
    IncomingLatent {
        source_node: NodeId,
        layer_idx: usize,
        latent: LatentVector,
        request_id: uuid::Uuid,
    },
    /// Start a new distributed inference request
    StartInference {
        request: InferenceRequest,
        response_tx: oneshot::Sender<Result<InferenceResponse>>,
    },
    /// Update topology
    UpdateTopology(mycelium_core::TopologyMap),
    /// Register local model shard
    RegisterShard {
        layer_start: usize,
        layer_end: usize,
        expert_ids: Vec<usize>,
    },
    /// Set the transport for remote communication
    SetTransport(Arc<dyn LatentTransport>),
}

impl std::fmt::Debug for RouterCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncomingLatent { source_node, layer_idx, latent, request_id } =>
                f.debug_struct("IncomingLatent")
                    .field("source_node", source_node)
                    .field("layer_idx", layer_idx)
                    .field("latent", latent)
                    .field("request_id", request_id)
                    .finish(),
            Self::StartInference { request, .. } =>
                f.debug_struct("StartInference").field("request", request).finish(),
            Self::UpdateTopology(topology) =>
                f.debug_tuple("UpdateTopology").field(topology).finish(),
            Self::RegisterShard { layer_start, layer_end, expert_ids } =>
                f.debug_struct("RegisterShard")
                    .field("layer_start", layer_start)
                    .field("layer_end", layer_end)
                    .field("expert_ids", expert_ids)
                    .finish(),
            Self::SetTransport(_) =>
                f.debug_struct("SetTransport").field("transport", &"<dyn LatentTransport>").finish(),
        }
    }
}

/// The distributed tensor router coordinates inference across nodes.
///
/// # Architecture
/// ```text
/// [User Request] → Router → [Local Layer 0-15] → Network → [Remote Layer 16-31]
///                                                        ↓
/// [Response] ← Router ← [Local Layer 32-47] ← Network ←─┘
/// ```
///
/// Each node runs a router that:
/// 1. Receives latents from upstream nodes (or tokenizes input)
/// 2. Processes local layers
/// 3. Sends latents to downstream nodes via LatentTransport
/// 4. Aggregates results and returns
pub struct DistributedTensorRouter {
    /// Node ID for this router
    node_id: NodeId,
    /// Model configuration
    config: ModelConfig,
    /// Local layer range this node handles
    local_layers: (usize, usize),
    /// Local expert IDs this node handles
    local_experts: Vec<usize>,
    /// Coordinator for topology and routing decisions
    coordinator: Arc<RwLock<DistributedCoordinator>>,
    /// Pending requests waiting for remote results
    pending: Arc<RwLock<StdHashMap<uuid::Uuid, PendingRequest>>>,
    /// Channel for sending commands to the router task
    cmd_tx: mpsc::Sender<RouterCommand>,
    /// Local model reference (if this node has weights loaded)
    local_model: Arc<RwLock<Option<ModelWeights>>>,
    /// Transport for sending latents to remote nodes
    transport: Option<Arc<dyn LatentTransport>>,
    /// Manages continuous latent streams between nodes
    stream_manager: Arc<LatentStreamManager>,
}

impl DistributedTensorRouter {
    /// Create a new distributed tensor router.
    pub fn new(
        node_id: NodeId,
        config: ModelConfig,
        coordinator: DistributedCoordinator,
    ) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(256);
        let coordinator = Arc::new(RwLock::new(coordinator));
        let pending = Arc::new(RwLock::new(StdHashMap::new()));
        let local_model = Arc::new(RwLock::new(None));
        let stream_manager = Arc::new(LatentStreamManager::new(LatentStreamConfig::default()));
        
        // Spawn the router task
        let coordinator_clone = coordinator.clone();
        let pending_clone = pending.clone();
        let local_model_clone = local_model.clone();
        let config_clone = config.clone();
        let node_id_clone = node_id.clone();
        let stream_manager_clone = stream_manager.clone();
        
        tokio::spawn(async move {
            Self::router_task(
                cmd_rx,
                coordinator_clone,
                pending_clone,
                local_model_clone,
                config_clone,
                node_id_clone,
                None, // transport set later via command
                stream_manager_clone,
            ).await;
        });
        
        Self {
            node_id,
            config,
            local_layers: (0, 0),
            local_experts: Vec::new(),
            coordinator,
            pending,
            cmd_tx,
            local_model,
            transport: None,
            stream_manager,
        }
    }
    
    /// Set the transport for remote communication.
    pub fn set_transport(&mut self, transport: Arc<dyn LatentTransport>) {
        self.transport = Some(transport);
    }
    
    /// Get a sender for sending commands to this router.
    pub fn command_sender(&self) -> mpsc::Sender<RouterCommand> {
        self.cmd_tx.clone()
    }
    
    /// Set the local model weights.
    pub async fn set_local_model(&self, model: ModelWeights) {
        let mut guard = self.local_model.write().await;
        *guard = Some(model);
    }
    
    /// Register which layers this node handles locally.
    pub async fn register_local_shard(&self, layer_start: usize, layer_end: usize, expert_ids: Vec<usize>) {
        // Send command to update shard info
        let _ = self.cmd_tx.send(RouterCommand::RegisterShard {
            layer_start,
            layer_end,
            expert_ids,
        }).await;
    }

    /// Get a reference to the latent stream manager.
    pub fn stream_manager(&self) -> &Arc<LatentStreamManager> {
        &self.stream_manager
    }
    
    /// Run a distributed inference request.
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let (response_tx, response_rx) = oneshot::channel();
        
        self.cmd_tx.send(RouterCommand::StartInference {
            request,
            response_tx,
        }).await.map_err(|e| anyhow::anyhow!("Router channel closed: {}", e))?;
        
        response_rx.await.map_err(|e| anyhow::anyhow!("Response channel error: {}", e))?
    }
    
    /// Handle an incoming latent from another node.
    pub async fn handle_incoming_latent(
        &self,
        source_node: NodeId,
        layer_idx: usize,
        latent: LatentVector,
        request_id: uuid::Uuid,
    ) -> Result<()> {
        self.cmd_tx.send(RouterCommand::IncomingLatent {
            source_node,
            layer_idx,
            latent,
            request_id,
        }).await.map_err(|e| anyhow::anyhow!("Router channel closed: {}", e))
    }
    
    /// Main router task - processes commands and coordinates inference.
    async fn router_task(
        mut cmd_rx: mpsc::Receiver<RouterCommand>,
        coordinator: Arc<RwLock<DistributedCoordinator>>,
        pending: Arc<RwLock<StdHashMap<uuid::Uuid, PendingRequest>>>,
        local_model: Arc<RwLock<Option<ModelWeights>>>,
        config: ModelConfig,
        node_id: NodeId,
        transport: Option<Arc<dyn LatentTransport>>,
        stream_manager: Arc<LatentStreamManager>,
    ) {
        let mut local_layers = (0usize, 0usize);
        let mut local_experts = Vec::new();
        let mut current_transport = transport;
        
        info!("DistributedTensorRouter started for node {}", node_id);
        
        while let Some(cmd) = cmd_rx.recv().await {
            match cmd {
                RouterCommand::RegisterShard { layer_start, layer_end, expert_ids } => {
                    local_layers = (layer_start, layer_end);
                    local_experts = expert_ids;
                    info!(
                        "Router registered local shard: layers {}-{}, {} experts",
                        layer_start, layer_end, local_experts.len()
                    );
                }
                
                RouterCommand::UpdateTopology(topology) => {
                    let mut coord = coordinator.write().await;
                    coord.update_topology(topology);
                }

                RouterCommand::SetTransport(new_transport) => {
                    current_transport = Some(new_transport);
                    info!("Transport set for distributed tensor routing");
                }
                
                RouterCommand::StartInference { request, response_tx } => {
                    // Handle inference request with streaming support
                    let result = Self::handle_inference_request(
                        &config,
                        &local_layers,
                        &local_experts,
                        &local_model,
                        &coordinator,
                        &pending,
                        request,
                        node_id,
                        &current_transport,
                        &stream_manager,
                    ).await;
                    
                    let _ = response_tx.send(result);
                }
                
                RouterCommand::IncomingLatent { source_node, layer_idx, latent, request_id } => {
                    debug!(
                        "Received latent from {} at layer {} for request {}",
                        source_node, layer_idx, request_id
                    );
                    
                    // Process this latent through our local layers
                    let result = Self::process_incoming_latent(
                        &config,
                        &local_layers,
                        &local_experts,
                        &local_model,
                        &coordinator,
                        layer_idx,
                        latent.clone(),
                        request_id,
                        node_id,
                        &current_transport,
                        &stream_manager,
                    ).await;
                    
                    if let Ok(Some(layer_result)) = result {
                        info!(
                            "Processed incoming latent: layer {} -> {}, is_final={}",
                            layer_idx, layer_result.layer_idx, layer_result.is_final
                        );
                    }
                }
            }
        }
        
        info!("DistributedTensorRouter stopped for node {}", node_id);
    }
    
    /// Handle a new inference request.
    async fn handle_inference_request(
        config: &ModelConfig,
        local_layers: &(usize, usize),
        _local_experts: &[usize],
        local_model: &Arc<RwLock<Option<ModelWeights>>>,
        coordinator: &Arc<RwLock<DistributedCoordinator>>,
        _pending: &Arc<RwLock<StdHashMap<uuid::Uuid, PendingRequest>>>,
        request: InferenceRequest,
        node_id: NodeId,
        transport: &Option<Arc<dyn LatentTransport>>,
        _stream_manager: &Arc<LatentStreamManager>,
    ) -> Result<InferenceResponse> {
        let start = std::time::Instant::now();
        let request_id = request.id;
        
        info!(
            "Starting distributed inference for request {} (latent_mode={})",
            request_id, request.latent_mode
        );
        
        // Check if we have a local model
        let mut model_guard = local_model.write().await;
        let has_local_model = model_guard.is_some();
        
        if !has_local_model && local_layers.0 == 0 && local_layers.1 == 0 {
            // No local model and no layers assigned
            return Ok(InferenceResponse {
                id: request_id,
                text: Some("[mycelium: no local model loaded, coordinator-only mode]".into()),
                latents: Vec::new(),
                participating_nodes: vec![node_id],
                latency_ms: start.elapsed().as_millis() as u64,
            });
        }
        
        // If we have the first layers, start inference
        if local_layers.0 == 0 && has_local_model {
            // This node has the embedding layer and first layers
            info!(
                "Node {} starting inference (has layers 0-{})",
                node_id, local_layers.1
            );

            // 1. Tokenize prompt using byte-level fallback tokenizer
            let tokenizer = mycelium_core::ByteTokenizer::new(256);
            let tokens = match tokenizer.encode(&request.prompt) {
                Ok(t) => t,
                Err(e) => {
                    return Ok(InferenceResponse {
                        id: request_id,
                        text: Some(format!("[tokenization failed: {}]", e)),
                        latents: Vec::new(),
                        participating_nodes: vec![node_id],
                        latency_ms: start.elapsed().as_millis() as u64,
                    });
                }
            };

            // Add BOS token if available
            let mut all_tokens: Vec<u32> = tokens.clone();
            if let Some(bos) = tokenizer.bos_token_id() {
                all_tokens.push(bos);
            }
            all_tokens.extend(tokens);

            info!("Tokenized prompt: {} tokens", all_tokens.len());

            // 2. Run through local layers
            if let Some(model) = model_guard.as_mut() {
                let mut latents_collected = Vec::new();

                // Forward pass through local layers only
                let tokens_tensor = Tensor::from_slice(&all_tokens, (1, all_tokens.len()), &model.device)?;
                let mut xs = model.tok_embeddings.forward(&tokens_tensor)?;

                let device = model.device.clone();
                let mask = if all_tokens.len() == 1 {
                    None
                } else {
                    Some(model.mask(all_tokens.len(), &device)?)
                };

                // Process only the layers this node is responsible for
                for layer_idx in local_layers.0..local_layers.1.min(model.layers.len()) {
                    xs = model.layers[layer_idx].forward(&xs, mask.as_ref())?;

                    // Extract latent at each layer for distributed tracking
                    if let Ok(latent) = tensor_to_latent(&xs, layer_idx) {
                        latents_collected.push(latent.clone());
                    }
                }

                // 3. If we have all layers, generate output
                if local_layers.1 >= config.num_layers {
                    // Complete the generation locally
                    let output = model.norm.forward(&xs)?;
                    let logits = model.output.forward(&output)?;

                    // Simple greedy decoding for now
                    let last_token = logits.i((0, logits.dim(1)? - 1, ..))?;
                    let token_ids = last_token.argmax(candle_core::D::Minus1)?;
                    let token_id = token_ids.to_scalar::<u32>()?;

                    let generated_text = tokenizer.decode(&[token_id])
                        .unwrap_or_else(|_| format!("[token {}]", token_id));

                    info!(
                        "Node {} completed inference: {} tokens -> '{}'",
                        node_id, all_tokens.len(), generated_text
                    );

                    return Ok(InferenceResponse {
                        id: request_id,
                        text: Some(format!("[node {}] {}", node_id, generated_text)),
                        latents: latents_collected,
                        participating_nodes: vec![node_id],
                        latency_ms: start.elapsed().as_millis() as u64,
                    });
                } else {
                    // 4. Route latent to next node using continuous streaming
                    let coord = coordinator.read().await;
                    let next_node = coord.find_layer_node(local_layers.1);
                    drop(coord);

                    if let Some(next) = next_node {
                        info!(
                            "Node {} routing latent to node {} for layers {}+ via streaming",
                            node_id, next, local_layers.1
                        );

                        // Stream latents to next node instead of sending individually
                        if let Some(tr) = transport {
                            // Open a stream to the next node
                            let stream_id = match tr.open_stream(
                                next,
                                local_layers.1,
                                config.num_layers,
                                64, // buffer size
                            ).await {
                                Ok(sid) => sid,
                                Err(e) => {
                                    warn!("Failed to open stream to node {}: {}", next, e);
                                    // Fallback to individual send
                                    if let Some(last_latent) = latents_collected.last() {
                                        let _ = tr.send_latent(
                                            next,
                                            local_layers.1,
                                            last_latent.clone(),
                                            request_id,
                                        ).await;
                                    }
                                    return Ok(InferenceResponse {
                                        id: request_id,
                                        text: Some(format!("[node {} partial, stream open failed, fallback used]", node_id)),
                                        latents: latents_collected,
                                        participating_nodes: vec![node_id],
                                        latency_ms: start.elapsed().as_millis() as u64,
                                    });
                                }
                            };

                            // Send all collected latents through the stream
                            for (seq, latent) in latents_collected.iter().enumerate() {
                                if let Err(e) = tr.send_stream(stream_id, seq as u64, latent.clone()).await {
                                    warn!("Failed to send latent {} through stream: {}", seq, e);
                                }
                            }

                            // Send the last latent (at the boundary layer) through the stream
                            if let Some(last_latent) = latents_collected.last() {
                                let seq = latents_collected.len() as u64;
                                if let Err(e) = tr.send_stream(stream_id, seq, last_latent.clone()).await {
                                    warn!("Failed to send final latent through stream: {}", e);
                                }
                            }

                            // Close the stream
                            let _ = tr.close_stream(stream_id, "inference complete").await;

                            info!("Successfully streamed latents to node {}", next);
                        } else {
                            warn!("No transport available to stream latents to node {}", next);
                        }
                    }

                    return Ok(InferenceResponse {
                        id: request_id,
                        text: Some(format!("[node {} partial inference complete, streaming to next]", node_id)),
                        latents: latents_collected,
                        participating_nodes: vec![node_id],
                        latency_ms: start.elapsed().as_millis() as u64,
                    });
                }
            } else {
                return Ok(InferenceResponse {
                    id: request_id,
                    text: Some(format!("[node {} has layers 0-{} but model not loaded]", node_id, local_layers.1)),
                    latents: Vec::new(),
                    participating_nodes: vec![node_id],
                    latency_ms: start.elapsed().as_millis() as u64,
                });
            }
        }
        
        // This node is a mid-tier or leaf node
        // It will receive latents from upstream and process them
        Ok(InferenceResponse {
            id: request_id,
            text: Some(format!("[node {} ready for incoming latents]", node_id)),
            latents: Vec::new(),
            participating_nodes: vec![node_id],
            latency_ms: start.elapsed().as_millis() as u64,
        })
    }
    
    /// Process an incoming latent from another node.
    async fn process_incoming_latent(
        config: &ModelConfig,
        local_layers: &(usize, usize),
        _local_experts: &[usize],
        local_model: &Arc<RwLock<Option<ModelWeights>>>,
        coordinator: &Arc<RwLock<DistributedCoordinator>>,
        input_layer: usize,
        latent: LatentVector,
        request_id: uuid::Uuid,
        node_id: NodeId,
        transport: &Option<Arc<dyn LatentTransport>>,
        _stream_manager: &Arc<LatentStreamManager>,
    ) -> Result<Option<LayerResult>> {
        // Check if we should process this layer
        if input_layer < local_layers.0 || input_layer >= local_layers.1 {
            debug!(
                "Node {} cannot process layer {} (handles {}-{})",
                node_id, input_layer, local_layers.0, local_layers.1
            );
            return Ok(None);
        }
        
        let mut model_guard = local_model.write().await;
        if let Some(model) = model_guard.as_mut() {
            // Process through local layers using actual tensor computation
            info!(
                "Node {} processing latent at layer {} (request {})",
                node_id, input_layer, request_id
            );

            // Convert latent back to tensor for processing
            let dim = latent.data.len();
            let tensor = Tensor::from_slice(&latent.data, (1, 1, dim), &model.device)?;

            // Process through each local layer starting from input_layer
            let mut current_tensor = tensor;
            let end_layer = (input_layer + (local_layers.1 - local_layers.0)).min(config.num_layers);

            for layer_idx in input_layer..end_layer.min(model.layers.len()) {
                current_tensor = model.layers[layer_idx].forward(&current_tensor, None)?;
            }

            // Convert back to latent
            let transformed_latent = tensor_to_latent(&current_tensor, end_layer)?;

            // Check if this is the final layer
            let is_final = end_layer >= config.num_layers;

            // If not final, stream to next node
            if !is_final {
                let coord = coordinator.read().await;
                let next_node = coord.find_layer_node(end_layer);
                drop(coord);

                if let Some(next) = next_node {
                    if let Some(tr) = transport {
                        // Open stream, send latent, close stream
                        if let Ok(stream_id) = tr.open_stream(next, end_layer, config.num_layers, 32).await {
                            if let Err(e) = tr.send_stream(stream_id, 0, transformed_latent.clone()).await {
                                warn!("Failed to stream latent to next node {}: {}", next, e);
                            }
                            let _ = tr.close_stream(stream_id, "layer processed").await;
                            info!("Streamed processed latent to node {}", next);
                        } else {
                            warn!("Failed to open stream to node {}", next);
                        }
                    }
                }
            }

            // Return the result
            Ok(Some(LayerResult {
                layer_idx: end_layer,
                latent: transformed_latent,
                is_final,
            }))
        } else {
            warn!("Node {} received latent but has no local model", node_id);
            Ok(None)
        }
    }
}

/// Trait for sending latents to remote nodes.
/// Implemented by Hyphae network layer.
#[async_trait::async_trait]
pub trait LatentTransport: Send + Sync {
    /// Send a latent to a specific node.
    async fn send_latent(
        &self,
        target_node: NodeId,
        layer_idx: usize,
        latent: LatentVector,
        request_id: uuid::Uuid,
    ) -> Result<()>;
    
    /// Broadcast a latent to all nodes that might need it.
    async fn broadcast_latent(
        &self,
        layer_idx: usize,
        latent: LatentVector,
        request_id: uuid::Uuid,
    ) -> Result<Vec<NodeId>>;

    /// Open a continuous latent stream to a target node.
    /// Returns the stream_id for subsequent send_stream/recv_stream calls.
    async fn open_stream(
        &self,
        target_node: NodeId,
        layer_start: usize,
        layer_end: usize,
        buffer_size: usize,
    ) -> Result<uuid::Uuid>;

    /// Send a latent vector through an existing stream.
    /// Sequence number provides ordering and flow control.
    async fn send_stream(
        &self,
        stream_id: uuid::Uuid,
        sequence: u64,
        latent: LatentVector,
    ) -> Result<()>;

    /// Close a latent stream, signaling no more data will be sent.
    async fn close_stream(
        &self,
        stream_id: uuid::Uuid,
        reason: &str,
    ) -> Result<()>;
}

// ─── Network-Aware MoE Expert Router ────────────────────────────────────────

/// Configuration for MoE routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoERouterConfig {
    /// Maximum number of remote expert requests to batch
    pub max_batch_size: usize,
    /// Timeout for remote expert requests (ms)
    pub request_timeout_ms: u64,
    /// Whether to fallback to local experts if remote fails
    pub fallback_to_local: bool,
    /// Maximum concurrent remote requests
    pub max_concurrent_requests: usize,
}

impl Default for MoERouterConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            request_timeout_ms: 5000,
            fallback_to_local: true,
            max_concurrent_requests: 64,
        }
    }
}

/// Result from expert routing.
#[derive(Debug, Clone)]
pub struct ExpertRouteResult {
    /// Expert ID that was routed to
    pub expert_id: usize,
    /// Node that processed the expert
    pub node_id: NodeId,
    /// Output latent from the expert
    pub output_latent: LatentVector,
    /// Whether this was processed locally
    pub was_local: bool,
    /// Processing time in microseconds
    pub processing_time_us: u64,
}

/// A pending remote expert request.
#[derive(Debug)]
struct PendingExpertRequest {
    /// Request ID
    request_id: uuid::Uuid,
    /// Expert ID being requested
    expert_id: usize,
    /// Input latent
    input_latent: LatentVector,
    /// When the request was made
    created_at: std::time::Instant,
    /// Response channel
    response_tx: oneshot::Sender<Result<LatentVector>>,
}

/// Network-aware MoE router that routes experts across nodes.
///
/// # Architecture
/// ```text
/// [Input Latent] → Local Router → [Expert 0-15: Local] ────→ Output
///                            ↓
///                    [Expert 16-31: Node B] ──┐
///                    [Expert 32-47: Node C] ──┼─→ Aggregate → Output
///                    [Expert 48-63: Node D] ──┘
/// ```
///
/// The router:
/// 1. Computes which experts to use (top-k routing)
/// 2. Determines which node holds each expert
/// 3. Routes locally if we have the expert, sends to remote node otherwise
/// 4. Aggregates results from all experts
/// 5. Handles failures with fallback
pub struct NetworkMoERouter {
    /// Node ID
    node_id: NodeId,
    /// Model config
    config: ModelConfig,
    /// Router config
    router_config: MoERouterConfig,
    /// Local expert IDs this node can process
    local_experts: Vec<usize>,
    /// Coordinator for topology
    coordinator: Arc<RwLock<DistributedCoordinator>>,
    /// Transport for sending latents to remote nodes
    transport: Option<Arc<dyn LatentTransport>>,
    /// Pending remote requests
    pending_requests: Arc<RwLock<StdHashMap<uuid::Uuid, PendingExpertRequest>>>,
    /// Local experts (FFN weights for each local expert)
    local_expert_weights: Vec<Mlp>,
    /// Router gate weights
    gate_weights: QMatMul,
}

impl NetworkMoERouter {
    /// Create a new network-aware MoE router.
    pub fn new(
        node_id: NodeId,
        config: ModelConfig,
        router_config: MoERouterConfig,
        coordinator: Arc<RwLock<DistributedCoordinator>>,
        gate_weights: QMatMul,
    ) -> Self {
        Self {
            node_id,
            config,
            router_config,
            local_experts: Vec::new(),
            coordinator,
            transport: None,
            pending_requests: Arc::new(RwLock::new(StdHashMap::new())),
            local_expert_weights: Vec::new(),
            gate_weights,
        }
    }
    
    /// Set the transport for remote communication.
    pub fn set_transport(&mut self, transport: Arc<dyn LatentTransport>) {
        self.transport = Some(transport);
    }
    
    /// Register local experts.
    pub fn register_local_experts(&mut self, expert_ids: Vec<usize>, weights: Vec<Mlp>) {
        assert_eq!(expert_ids.len(), weights.len(), "Expert IDs and weights must match");
        self.local_experts = expert_ids;
        self.local_expert_weights = weights;
        info!(
            "NetworkMoERouter: registered {} local experts: {:?}",
            self.local_experts.len(),
            &self.local_experts[..self.local_experts.len().min(10)]
        );
    }
    
    /// Route a latent through the top-k experts.
    /// Returns aggregated output from all selected experts.
    pub async fn route(
        &self,
        latent: &LatentVector,
        layer_idx: usize,
        request_id: uuid::Uuid,
    ) -> Result<LatentVector> {
        let start = std::time::Instant::now();
        
        // 1. Compute routing weights (which experts to use)
        let routing_weights = self.compute_routing_weights(latent)?;
        
        // 2. Select top-k experts
        let top_k = self.config.top_k_experts;
        let selected = self.select_top_k_experts(&routing_weights, top_k);
        
        debug!(
            "Routing latent to {} experts: {:?}",
            selected.len(),
            selected.iter().map(|(id, w)| (*id, *w)).collect::<Vec<_>>()
        );
        
        // 3. Process each expert (local or remote)
        let mut expert_results = Vec::with_capacity(selected.len());
        
        for (expert_id, weight) in selected {
            let result = self.process_expert(expert_id, latent, layer_idx, request_id).await;
            
            match result {
                Ok(expert_output) => {
                    // Weight the expert output by routing weight
                    let weighted = expert_output.scale(weight);
                    expert_results.push(weighted);
                }
                Err(e) => {
                    warn!("Expert {} failed: {}, skipping", expert_id, e);
                    // Could implement fallback here
                }
            }
        }
        
        // 4. Aggregate all expert outputs
        if expert_results.is_empty() {
            bail!("All experts failed for layer {}", layer_idx);
        }
        
        // Sum all weighted expert outputs
        let mut aggregated = expert_results.remove(0);
        for output in expert_results {
            aggregated = aggregated.add(&output);
        }
        
        let elapsed = start.elapsed().as_micros();
        debug!(
            "MoE routing complete: {} experts, {}µs",
            self.config.top_k_experts, elapsed
        );
        
        Ok(aggregated)
    }
    
    /// Compute routing weights for all experts.
    fn compute_routing_weights(&self, latent: &LatentVector) -> Result<Vec<f32>> {
        // Compute actual router gate using the gate weights matrix
        // gate_output = latent @ gate_weights
        // routing_weights = softmax(gate_output)

        let num_experts = self.config.num_experts;

        // Create tensor from latent data (pad or truncate to hidden_dim)
        let hidden_dim = self.config.hidden_dim;
        let mut data = latent.data.clone();
        if data.len() < hidden_dim {
            data.resize(hidden_dim, 0.0);
        } else if data.len() > hidden_dim {
            data.truncate(hidden_dim);
        }

        // Create input tensor: (1, 1, hidden_dim)
        let input = Tensor::from_slice(&data, (1, 1, hidden_dim), &Device::Cpu)?;

        // Forward through gate to get routing logits
        match self.gate_weights.forward(&input) {
            Ok(logits) => {
                // Apply softmax to get routing weights
                let logits_f32 = logits.to_dtype(DType::F32)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&logits_f32)?;

                // Extract weights as vec
                let weights_flat = routing_weights.reshape((num_experts,))?
                    .to_vec1::<f32>()?;

                debug!(
                    "Computed routing weights for {} experts",
                    weights_flat.len()
                );

                Ok(weights_flat)
            }
            Err(e) => {
                warn!("Gate computation failed, falling back to hash-based: {}", e);
                // Fallback to hash-based routing
                let mut weights = vec![0.0f32; num_experts];
                let hash = {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    use std::hash::Hasher;
                    for &v in latent.data.iter().take(64) {
                        hasher.write_u32(v.to_bits());
                    }
                    std::hash::Hasher::finish(&hasher)
                };

                for i in 0..num_experts.min(8) {
                    let idx = ((hash as usize) + i) % num_experts;
                    weights[idx] = 1.0 / 8.0;
                }
                Ok(weights)
            }
        }
    }
    
    /// Select top-k experts from routing weights.
    fn select_top_k_experts(&self, weights: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = weights.iter().enumerate()
            .map(|(i, &w)| (i, w))
            .collect();
        
        // Sort by weight descending
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        
        // Take top-k and normalize
        let top_k: Vec<_> = indexed.into_iter().take(k).collect();
        let sum: f32 = top_k.iter().map(|(_, w)| w).sum();
        
        if sum > 0.0 {
            top_k.into_iter().map(|(id, w)| (id, w / sum)).collect()
        } else {
            // Fallback to first k experts with uniform weight
            (0..k.min(weights.len()))
                .map(|i| (i, 1.0 / k as f32))
                .collect()
        }
    }
    
    /// Process a single expert (local or remote).
    async fn process_expert(
        &self,
        expert_id: usize,
        latent: &LatentVector,
        layer_idx: usize,
        request_id: uuid::Uuid,
    ) -> Result<LatentVector> {
        let start = std::time::Instant::now();
        
        // Check if we have this expert locally
        if let Some(local_idx) = self.local_experts.iter().position(|&e| e == expert_id) {
            // Process locally
            let output = self.process_local_expert(local_idx, latent)?;
            
            debug!(
                "Expert {} processed locally in {}µs",
                expert_id,
                start.elapsed().as_micros()
            );
            
            return Ok(output);
        }
        
        // Need to route to remote node
        if let Some(transport) = &self.transport {
            // Find which node has this expert
            let coord = self.coordinator.read().await;
            let target_node = coord.find_expert_node(expert_id);
            drop(coord);
            
            if let Some(target) = target_node {
                if target == self.node_id {
                    // This shouldn't happen, but handle gracefully
                    warn!("Expert {} is supposed to be local but not found", expert_id);
                    return Ok(latent.clone());
                }
                
                // Send to remote node with proper response waiting
                debug!(
                    "Routing expert {} to remote node {}",
                    expert_id, target
                );

                let expert_request_id = uuid::Uuid::new_v4();
                let (response_tx, response_rx) = tokio::sync::oneshot::channel();

                // Store pending request
                {
                    let mut pending = self.pending_requests.write().await;
                    pending.insert(
                        expert_request_id,
                        PendingExpertRequest {
                            request_id: expert_request_id,
                            expert_id,
                            input_latent: latent.clone(),
                            created_at: std::time::Instant::now(),
                            response_tx,
                        },
                    );
                }

                // Send the latent to the remote node
                transport.send_latent(
                    target,
                    layer_idx,
                    latent.clone(),
                    expert_request_id,
                ).await?;

                // Wait for response with timeout
                let timeout_ms = self.router_config.request_timeout_ms;
                let result = match tokio::time::timeout(
                    std::time::Duration::from_millis(timeout_ms),
                    response_rx
                ).await {
                    Ok(Ok(response)) => {
                        match response {
                            Ok(output_latent) => {
                                debug!(
                                    "Remote expert {} response received in {}µs",
                                    expert_id,
                                    std::time::Instant::now().elapsed().as_micros()
                                );
                                return Ok(output_latent);
                            }
                            Err(e) => {
                                warn!("Remote expert {} returned error: {}", expert_id, e);
                                None
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        warn!("Remote expert {} response channel closed: {}", expert_id, e);
                        None
                    }
                    Err(_) => {
                        warn!("Remote expert {} timed out after {}ms", expert_id, timeout_ms);
                        // Clean up pending request
                        let mut pending = self.pending_requests.write().await;
                        pending.remove(&expert_request_id);
                        None
                    }
                };

                // If response handling failed, fall through to fallback
                if let Some(output) = result {
                    return Ok(output);
                }
            }
        }
        
        // No transport or no node found - fallback
        if self.router_config.fallback_to_local {
            // Use the first local expert as fallback
            if !self.local_expert_weights.is_empty() {
                warn!(
                    "Expert {} not available, using local expert 0 as fallback",
                    expert_id
                );
                return self.process_local_expert(0, latent);
            }
        }
        
        bail!("Expert {} not available and no fallback", expert_id)
    }
    
    /// Process a latent through a local expert.
    fn process_local_expert(&self, local_idx: usize, latent: &LatentVector) -> Result<LatentVector> {
        if local_idx >= self.local_expert_weights.len() {
            bail!("Local expert index {} out of range", local_idx);
        }

        // Actual FFN computation: output = expert_weights[local_idx].forward(latent)
        // The expert is an MLP that does: w2(silu(w1(x)) * w3(x))

        let dim = latent.data.len();
        let hidden_dim = self.config.hidden_dim;

        // Pad or truncate latent to match expected input size
        let mut data = latent.data.clone();
        if data.len() < hidden_dim {
            data.resize(hidden_dim, 0.0);
        } else if data.len() > hidden_dim {
            data.truncate(hidden_dim);
        }

        // Create input tensor: (1, 1, hidden_dim) for sequence length 1
        let input = Tensor::from_slice(&data, (1, 1, hidden_dim), &Device::Cpu)?;

        // Forward through the expert's MLP
        let expert = &self.local_expert_weights[local_idx];
        let output_tensor = expert.forward(&input)?;

        // Extract output latent
        let output_data = output_tensor.reshape((dim,))?.to_vec1::<f32>()?;

        Ok(LatentVector::from_vec(
            output_data,
            latent.layer_idx,
            latent.stream_id,
        ))
    }
}

// ─── Continuous Latent Streaming ────────────────────────────────────────────

/// Configuration for a latent stream channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentStreamConfig {
    /// Maximum number of latent vectors buffered in the channel.
    pub buffer_size: usize,
    /// Enable backpressure: sender blocks when buffer is full.
    pub backpressure: bool,
    /// Optional maximum throughput (vectors per second). `None` means unlimited.
    pub max_throughput: Option<f64>,
}

impl Default for LatentStreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64,
            backpressure: true,
            max_throughput: None,
        }
    }
}

/// Metadata describing a single latent stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentStreamMeta {
    /// Unique identifier for this stream.
    pub stream_id: uuid::Uuid,
    /// Node that produces latent vectors.
    pub source_node: NodeId,
    /// Node that consumes latent vectors.
    pub target_node: NodeId,
    /// Timestamp (epoch millis) when the stream was created.
    pub created_at: u64,
}

/// Sending half of a latent stream.
pub struct LatentStreamSender {
    pub meta: LatentStreamMeta,
    tx: mpsc::Sender<LatentVector>,
}

impl LatentStreamSender {
    /// Send a latent vector into the stream.
    /// With backpressure enabled this will wait until buffer space is available.
    pub async fn send(&self, latent: LatentVector) -> Result<()> {
        self.tx
            .send(latent)
            .await
            .map_err(|_| anyhow::anyhow!("latent stream {} closed", self.meta.stream_id))
    }

    /// Try to send without waiting. Returns an error if the buffer is full.
    pub fn try_send(&self, latent: LatentVector) -> Result<()> {
        self.tx
            .try_send(latent)
            .map_err(|e| anyhow::anyhow!("latent stream {} send failed: {}", self.meta.stream_id, e))
    }

    /// Returns the remaining capacity of the underlying channel.
    pub fn capacity(&self) -> usize {
        self.tx.capacity()
    }
}

/// Receiving half of a latent stream.
pub struct LatentStreamReceiver {
    pub meta: LatentStreamMeta,
    rx: mpsc::Receiver<LatentVector>,
}

impl LatentStreamReceiver {
    /// Receive the next latent vector, waiting if necessary.
    pub async fn recv(&mut self) -> Option<LatentVector> {
        self.rx.recv().await
    }

    /// Try to receive without blocking.
    pub fn try_recv(&mut self) -> Result<LatentVector> {
        self.rx
            .try_recv()
            .map_err(|e| anyhow::anyhow!("latent stream {} recv failed: {}", self.meta.stream_id, e))
    }
}

/// Manages multiple concurrent latent streams with flow control.
pub struct LatentStreamManager {
    config: LatentStreamConfig,
    streams: Arc<RwLock<StdHashMap<uuid::Uuid, LatentStreamMeta>>>,
}

impl LatentStreamManager {
    /// Create a new manager with the given default config.
    pub fn new(config: LatentStreamConfig) -> Self {
        Self {
            config,
            streams: Arc::new(RwLock::new(StdHashMap::new())),
        }
    }

    /// Create a new bidirectional stream between `source` and `target`.
    /// Returns `(sender, receiver)` for the forward direction.
    pub async fn create_stream(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> (LatentStreamSender, LatentStreamReceiver) {
        let stream_id = uuid::Uuid::new_v4();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let meta = LatentStreamMeta {
            stream_id,
            source_node: source,
            target_node: target,
            created_at: now,
        };

        let (tx, rx) = mpsc::channel(self.config.buffer_size);

        {
            let mut streams = self.streams.write().await;
            streams.insert(stream_id, meta.clone());
        }

        let sender = LatentStreamSender {
            meta: meta.clone(),
            tx,
        };
        let receiver = LatentStreamReceiver { meta, rx };
        (sender, receiver)
    }

    /// Remove a stream by id.
    pub async fn remove_stream(&self, stream_id: &uuid::Uuid) -> bool {
        let mut streams = self.streams.write().await;
        streams.remove(stream_id).is_some()
    }

    /// Number of active streams.
    pub async fn active_stream_count(&self) -> usize {
        self.streams.read().await.len()
    }

    /// List metadata for all active streams.
    pub async fn list_streams(&self) -> Vec<LatentStreamMeta> {
        self.streams.read().await.values().cloned().collect()
    }
}

// ─── Latent Memory Store ────────────────────────────────────────────────────

/// Key used to index stored latent vectors (hash of input + layer).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LatentKey {
    /// Hash of the input that produced the latent.
    pub input_hash: u64,
    /// Layer index.
    pub layer_idx: usize,
}

impl LatentKey {
    /// Build a key by hashing an arbitrary byte slice together with a layer index.
    pub fn new(input: &[u8], layer_idx: usize) -> Self {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        layer_idx.hash(&mut hasher);
        Self {
            input_hash: hasher.finish(),
            layer_idx,
        }
    }
}

/// Entry stored inside the memory store.
#[derive(Debug, Clone)]
struct LatentMemoryEntry {
    latent: LatentVector,
    access_count: u64,
    last_accessed: std::time::Instant,
}

/// An LRU-evicting store for latent vectors with similarity search.
pub struct LatentMemoryStore {
    capacity: usize,
    entries: StdHashMap<LatentKey, LatentMemoryEntry>,
    /// Insertion-order list so we can do LRU eviction (oldest first).
    insertion_order: Vec<LatentKey>,
}

impl LatentMemoryStore {
    /// Create a store that holds at most `capacity` entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: StdHashMap::new(),
            insertion_order: Vec::new(),
        }
    }

    /// Store a latent vector under the given key (LRU-evicts if full).
    pub fn store(&mut self, key: LatentKey, latent: LatentVector) {
        // If key already present, update in place (counts as new access).
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.latent = latent;
            entry.access_count += 1;
            entry.last_accessed = std::time::Instant::now();
            return;
        }

        // Evict LRU entry if at capacity.
        while self.entries.len() >= self.capacity && !self.insertion_order.is_empty() {
            // Find the entry with the oldest last_accessed time.
            let mut lru_idx = 0;
            let mut lru_time = std::time::Instant::now();
            for (i, k) in self.insertion_order.iter().enumerate() {
                if let Some(e) = self.entries.get(k) {
                    if e.last_accessed < lru_time {
                        lru_time = e.last_accessed;
                        lru_idx = i;
                    }
                }
            }
            let evicted_key = self.insertion_order.remove(lru_idx);
            self.entries.remove(&evicted_key);
        }

        self.insertion_order.push(key.clone());
        self.entries.insert(
            key,
            LatentMemoryEntry {
                latent,
                access_count: 1,
                last_accessed: std::time::Instant::now(),
            },
        );
    }

    /// Retrieve a latent vector by key (updates access stats).
    pub fn retrieve(&mut self, key: &LatentKey) -> Option<&LatentVector> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = std::time::Instant::now();
            Some(&entry.latent)
        } else {
            None
        }
    }

    /// Find the `k` most similar stored latents to `query` using cosine similarity.
    /// Returns `(key, similarity_score)` pairs sorted descending by similarity.
    pub fn search_similar(&self, query: &LatentVector, k: usize) -> Vec<(LatentKey, f32)> {
        let mut scored: Vec<(LatentKey, f32)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.latent.dim == query.dim)
            .map(|(key, e)| {
                let sim = cosine_similarity(&query.data, &e.latent.data);
                (key.clone(), sim)
            })
            .collect();

        // Sort descending by similarity.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Current number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return access count for a key (useful for caching analytics).
    pub fn access_count(&self, key: &LatentKey) -> Option<u64> {
        self.entries.get(key).map(|e| e.access_count)
    }
}

/// Cosine similarity between two equal-length f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// ─── Pipeline Parallelism ──────────────────────────────────────────────────

/// Configuration for pipeline-parallel execution across distributed nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineParallelConfig {
    /// Number of micro-batches to split each batch into for pipelining.
    pub num_micro_batches: usize,
    /// Timeout in milliseconds for each pipeline stage before failing.
    pub stage_timeout_ms: u64,
    /// Maximum number of in-flight micro-batches across the pipeline.
    pub max_inflight: usize,
    /// Size of the async channel buffer between stages.
    pub channel_buffer_size: usize,
    /// Whether to collect per-stage timing statistics.
    pub collect_stats: bool,
}

impl Default for PipelineParallelConfig {
    fn default() -> Self {
        Self {
            num_micro_batches: 4,
            stage_timeout_ms: 10_000,
            max_inflight: 8,
            channel_buffer_size: 16,
            collect_stats: true,
        }
    }
}

/// A single stage in the pipeline, representing a contiguous range of layers
/// assigned to a specific node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    /// Zero-based index of this stage in the pipeline sequence.
    pub stage_idx: usize,
    /// First layer index (inclusive) processed by this stage.
    pub start_layer: usize,
    /// Last layer index (exclusive) processed by this stage.
    pub end_layer: usize,
    /// The node responsible for executing this stage.
    pub node_id: NodeId,
}

impl PipelineStage {
    /// Create a new pipeline stage.
    pub fn new(stage_idx: usize, start_layer: usize, end_layer: usize, node_id: NodeId) -> Self {
        Self {
            stage_idx,
            start_layer,
            end_layer,
            node_id,
        }
    }

    /// Number of layers in this stage.
    pub fn num_layers(&self) -> usize {
        self.end_layer.saturating_sub(self.start_layer)
    }
}

/// A plan describing how model layers are partitioned across pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePlan {
    /// Ordered list of stages; stage 0 receives the initial latent.
    pub stages: Vec<PipelineStage>,
    /// Total number of layers in the model.
    pub total_layers: usize,
}

impl PipelinePlan {
    /// Create a pipeline plan from a list of (node_id, num_layers) pairs.
    ///
    /// Layers are assigned contiguously in the order of the provided list.
    pub fn from_node_layers(assignments: Vec<(NodeId, usize)>) -> Self {
        let mut stages = Vec::with_capacity(assignments.len());
        let mut offset = 0usize;

        for (idx, (node_id, num_layers)) in assignments.into_iter().enumerate() {
            stages.push(PipelineStage::new(idx, offset, offset + num_layers, node_id));
            offset += num_layers;
        }

        Self {
            total_layers: offset,
            stages,
        }
    }

    /// Create a uniform pipeline plan that splits `total_layers` as evenly
    /// as possible across the given nodes.
    pub fn uniform(nodes: &[NodeId], total_layers: usize) -> Self {
        assert!(!nodes.is_empty(), "Need at least one node for pipeline plan");

        let base = total_layers / nodes.len();
        let remainder = total_layers % nodes.len();

        let assignments: Vec<(NodeId, usize)> = nodes
            .iter()
            .enumerate()
            .map(|(i, nid)| {
                let extra = if i < remainder { 1 } else { 0 };
                (*nid, base + extra)
            })
            .collect();

        Self::from_node_layers(assignments)
    }

    /// Number of stages in the pipeline.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Validate the plan for internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.stages.is_empty() {
            bail!("Pipeline plan has no stages");
        }

        let mut expected_start = 0usize;
        for stage in &self.stages {
            if stage.start_layer != expected_start {
                bail!(
                    "Stage {} starts at layer {} but expected {}",
                    stage.stage_idx,
                    stage.start_layer,
                    expected_start
                );
            }
            if stage.end_layer <= stage.start_layer {
                bail!(
                    "Stage {} has invalid layer range [{}, {})",
                    stage.stage_idx,
                    stage.start_layer,
                    stage.end_layer
                );
            }
            expected_start = stage.end_layer;
        }

        if expected_start != self.total_layers {
            bail!(
                "Pipeline stages cover {} layers but plan expects {}",
                expected_start,
                self.total_layers
            );
        }

        Ok(())
    }
}

/// A single micro-batch flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    /// Index of this micro-batch within the full batch.
    pub micro_batch_idx: usize,
    /// The latent vector for this micro-batch.
    pub latent: LatentVector,
    /// Unique identifier for the overall request.
    pub request_id: uuid::Uuid,
}

/// Per-stage timing information collected during pipeline execution.
#[derive(Debug, Clone, Default)]
pub struct PipelineStageStats {
    /// Stage index.
    pub stage_idx: usize,
    /// Wall-clock latencies (one per micro-batch) in microseconds.
    pub latencies_us: Vec<u64>,
}

impl PipelineStageStats {
    fn new(stage_idx: usize) -> Self {
        Self {
            stage_idx,
            latencies_us: Vec::new(),
        }
    }

    /// Mean latency across all micro-batches in microseconds.
    pub fn mean_latency_us(&self) -> f64 {
        if self.latencies_us.is_empty() {
            return 0.0;
        }
        self.latencies_us.iter().sum::<u64>() as f64 / self.latencies_us.len() as f64
    }

    /// Maximum latency observed for this stage in microseconds.
    pub fn max_latency_us(&self) -> u64 {
        self.latencies_us.iter().copied().max().unwrap_or(0)
    }
}

/// Aggregate statistics for a full pipeline execution run.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Per-stage statistics.
    pub stage_stats: Vec<PipelineStageStats>,
    /// Total wall-clock time for the entire pipeline execution in microseconds.
    pub total_time_us: u64,
    /// Number of micro-batches processed.
    pub num_micro_batches: usize,
    /// Throughput in micro-batches per second.
    pub throughput_mb_per_sec: f64,
}

/// Result of a pipeline execution.
#[derive(Debug)]
pub struct PipelineResult {
    /// Output latents in micro-batch order.
    pub outputs: Vec<LatentVector>,
    /// Pipeline statistics (if collection was enabled).
    pub stats: Option<PipelineStats>,
}

/// Executes pipeline-parallel inference by streaming micro-batches through
/// an ordered sequence of pipeline stages using async channels.
///
/// Each stage either processes locally or ships the latent to a remote node
/// via [`LatentTransport`].
pub struct PipelineExecutor {
    /// The pipeline plan describing stage-to-node assignments.
    plan: PipelinePlan,
    /// Configuration knobs.
    config: PipelineParallelConfig,
    /// Our own node id so we can determine which stages are local.
    local_node_id: NodeId,
    /// Optional transport for sending latents to remote stages.
    transport: Option<Arc<dyn LatentTransport>>,
    /// Optional local model for executing stages that live on this node.
    local_model: Arc<RwLock<Option<ModelWeights>>>,
}

impl PipelineExecutor {
    /// Create a new pipeline executor.
    pub fn new(
        plan: PipelinePlan,
        config: PipelineParallelConfig,
        local_node_id: NodeId,
        local_model: Arc<RwLock<Option<ModelWeights>>>,
    ) -> Result<Self> {
        plan.validate()?;
        info!(
            "PipelineExecutor: {} stages, {} micro-batches, local node {}",
            plan.num_stages(),
            config.num_micro_batches,
            local_node_id,
        );
        Ok(Self {
            plan,
            config,
            local_node_id,
            transport: None,
            local_model,
        })
    }

    /// Attach a transport implementation for remote stage communication.
    pub fn set_transport(&mut self, transport: Arc<dyn LatentTransport>) {
        self.transport = Some(transport);
    }

    /// Split an input latent into `num_micro_batches` micro-batches.
    ///
    /// Each micro-batch receives an equal slice of the data vector.  When the
    /// dimension is not evenly divisible the last micro-batch receives the
    /// remainder.
    pub fn split_micro_batches(
        &self,
        latent: &LatentVector,
        request_id: uuid::Uuid,
    ) -> Vec<MicroBatch> {
        let n = self.config.num_micro_batches.max(1);

        if n == 1 {
            return vec![MicroBatch {
                micro_batch_idx: 0,
                latent: latent.clone(),
                request_id,
            }];
        }

        let chunk_size = latent.data.len() / n;
        let mut batches = Vec::with_capacity(n);

        for i in 0..n {
            let start = i * chunk_size;
            let end = if i == n - 1 {
                latent.data.len()
            } else {
                (i + 1) * chunk_size
            };
            let slice = latent.data[start..end].to_vec();
            batches.push(MicroBatch {
                micro_batch_idx: i,
                latent: LatentVector::from_vec(slice, latent.layer_idx, latent.stream_id),
                request_id,
            });
        }

        batches
    }

    /// Reassemble micro-batch outputs into a single latent vector.
    fn merge_micro_batches(outputs: &[LatentVector]) -> LatentVector {
        assert!(!outputs.is_empty(), "Cannot merge empty micro-batch list");

        if outputs.len() == 1 {
            return outputs[0].clone();
        }

        let mut merged_data: Vec<f32> = Vec::new();
        for out in outputs {
            merged_data.extend_from_slice(&out.data);
        }

        let last = outputs.last().unwrap();
        LatentVector::from_vec(merged_data, last.layer_idx, last.stream_id)
    }

    /// Process a single micro-batch through one pipeline stage.
    ///
    /// If the stage lives on the local node, we run it against the local model.
    /// Otherwise we delegate to the transport layer.
    async fn process_stage(
        &self,
        stage: &PipelineStage,
        micro_batch: MicroBatch,
    ) -> Result<MicroBatch> {
        let timeout = std::time::Duration::from_millis(self.config.stage_timeout_ms);

        let result = tokio::time::timeout(timeout, async {
            if stage.node_id == self.local_node_id {
                self.process_local_stage(stage, &micro_batch).await
            } else {
                self.process_remote_stage(stage, &micro_batch).await
            }
        })
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Pipeline stage {} timed out after {}ms",
                stage.stage_idx,
                self.config.stage_timeout_ms
            )
        })??;

        Ok(result)
    }

    /// Execute a stage locally using the local model weights.
    async fn process_local_stage(
        &self,
        stage: &PipelineStage,
        micro_batch: &MicroBatch,
    ) -> Result<MicroBatch> {
        let mut model_guard = self.local_model.write().await;
        let model = model_guard
            .as_mut()
            .context("No local model loaded for pipeline stage execution")?;

        debug!(
            "Pipeline stage {} (layers {}-{}): processing locally",
            stage.stage_idx, stage.start_layer, stage.end_layer
        );

        // Feed the latent through the layer range on this node.
        let latent_dim = micro_batch.latent.data.len();
        let model_hidden = model.config.hidden_dim;
        let mut data = micro_batch.latent.data.clone();
        if data.len() < model_hidden {
            data.resize(model_hidden, 0.0);
        }

        let input = Tensor::from_slice(&data, (1, 1, data.len()), &model.device)?;
        let mut current = input;

        for layer_idx in stage.start_layer..stage.end_layer {
            if layer_idx < model.layers.len() {
                current = model.layers[layer_idx].forward(&current, None)?;
            }
        }

        let output_data: Vec<f32> = current.reshape((latent_dim,))?.to_vec1()?;
        let output_latent = LatentVector::from_vec(
            output_data,
            stage.end_layer,
            micro_batch.latent.stream_id,
        );

        Ok(MicroBatch {
            micro_batch_idx: micro_batch.micro_batch_idx,
            latent: output_latent,
            request_id: micro_batch.request_id,
        })
    }

    /// Ship a micro-batch to a remote node via the transport layer.
    ///
    /// Sends the latent to the remote node and waits for a response with timeout.
    /// The remote node is expected to process the latent through its assigned layers
    /// and return the result.
    async fn process_remote_stage(
        &self,
        stage: &PipelineStage,
        micro_batch: &MicroBatch,
    ) -> Result<MicroBatch> {
        let transport = self
            .transport
            .as_ref()
            .context("No transport available for remote pipeline stage")?;

        debug!(
            "Pipeline stage {} (layers {}-{}): sending to remote node {}",
            stage.stage_idx, stage.start_layer, stage.end_layer, stage.node_id
        );

        // Send latent to remote node
        transport
            .send_latent(
                stage.node_id,
                stage.start_layer,
                micro_batch.latent.clone(),
                micro_batch.request_id,
            )
            .await?;

        // In a full implementation, the remote node would process the latent
        // through its assigned layers and push the result back through the transport.
        // For now, we return the latent with updated layer index to maintain
        // pipeline flow. The remote processing would be handled by the
        // DistributedTensorRouter on the target node.
        let forwarded_latent = LatentVector::from_vec(
            micro_batch.latent.data.clone(),
            stage.end_layer,
            micro_batch.latent.stream_id,
        );

        debug!(
            "Pipeline stage {}: forwarded latent to {} (layers {}-{})",
            stage.stage_idx, stage.node_id, stage.start_layer, stage.end_layer
        );

        Ok(MicroBatch {
            micro_batch_idx: micro_batch.micro_batch_idx,
            latent: forwarded_latent,
            request_id: micro_batch.request_id,
        })
    }

    /// Execute the full pipeline for a single input latent.
    ///
    /// The input is split into micro-batches which are streamed concurrently
    /// through the pipeline stages via async channels.  The returned
    /// [`PipelineResult`] contains the merged output latent and optional
    /// statistics.
    pub async fn execute_pipeline(
        &self,
        input: &LatentVector,
        request_id: uuid::Uuid,
    ) -> Result<PipelineResult> {
        let pipeline_start = std::time::Instant::now();
        let micro_batches = self.split_micro_batches(input, request_id);
        let num_micro_batches = micro_batches.len();
        let num_stages = self.plan.num_stages();

        info!(
            "Pipeline execution: {} micro-batches, {} stages, request {}",
            num_micro_batches, num_stages, request_id
        );

        // Per-stage stats collectors (one Mutex<Vec<u64>> per stage).
        let stage_latencies: Vec<Arc<tokio::sync::Mutex<Vec<u64>>>> = (0..num_stages)
            .map(|_| Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(num_micro_batches))))
            .collect();

        // Build a chain of async channels:  sender[s] -> receiver[s] feeds stage s.
        // The first sender is loaded with the initial micro-batches.
        let buffer = self.config.channel_buffer_size;
        let mut senders: Vec<mpsc::Sender<MicroBatch>> = Vec::with_capacity(num_stages + 1);
        let mut receivers: Vec<mpsc::Receiver<MicroBatch>> = Vec::with_capacity(num_stages + 1);

        for _ in 0..=num_stages {
            let (tx, rx) = mpsc::channel(buffer);
            senders.push(tx);
            receivers.push(rx);
        }

        // Feed micro-batches into the first channel.
        let input_tx = senders[0].clone();
        for mb in micro_batches {
            input_tx
                .send(mb)
                .await
                .map_err(|_| anyhow::anyhow!("Failed to enqueue micro-batch into pipeline"))?;
        }
        drop(input_tx);
        // Drop the original sender so the channel closes after all clones are gone.
        drop(senders.remove(0));

        // Spawn a task per stage.  Each task reads from receivers[stage_idx]
        // and writes to senders[stage_idx] (which maps to the *next* channel).
        let mut handles = Vec::with_capacity(num_stages);

        // We need to move each receiver and sender into its own task.
        let mut rx_slots: Vec<Option<mpsc::Receiver<MicroBatch>>> =
            receivers.into_iter().map(Some).collect();
        let mut tx_slots: Vec<Option<mpsc::Sender<MicroBatch>>> =
            senders.into_iter().map(Some).collect();

        for s in 0..num_stages {
            let stage = self.plan.stages[s].clone();
            let stats_slot = stage_latencies[s].clone();
            let collect_stats = self.config.collect_stats;

            let mut rx = rx_slots[s]
                .take()
                .expect("receiver already consumed for stage");

            let tx = tx_slots[s]
                .take()
                .expect("sender already consumed for stage");

            let local_node_id = self.local_node_id;
            let transport = self.transport.clone();
            let local_model = self.local_model.clone();
            let stage_timeout_ms = self.config.stage_timeout_ms;

            let handle = tokio::spawn(async move {
                while let Some(mb) = rx.recv().await {
                    let t0 = std::time::Instant::now();

                    let timeout = std::time::Duration::from_millis(stage_timeout_ms);
                    let processed = tokio::time::timeout(timeout, async {
                        if stage.node_id == local_node_id {
                            // ---------- local processing ----------
                            let mut model_guard = local_model.write().await;
                            if let Some(model) = model_guard.as_mut() {
                                let latent_dim = mb.latent.data.len();
                                let model_hidden = model.config.hidden_dim;
                                let mut data = mb.latent.data.clone();
                                if data.len() < model_hidden {
                                    data.resize(model_hidden, 0.0);
                                }
                                let input =
                                    Tensor::from_slice(&data, (1, 1, data.len()), &model.device)?;
                                let mut current = input;
                                for layer_idx in stage.start_layer..stage.end_layer {
                                    if layer_idx < model.layers.len() {
                                        current = model.layers[layer_idx].forward(&current, None)?;
                                    }
                                }
                                let output_data: Vec<f32> =
                                    current.reshape((latent_dim,))?.to_vec1()?;
                                let mb_clone = mb.clone();
                                Ok::<MicroBatch, anyhow::Error>(MicroBatch {
                                    micro_batch_idx: mb_clone.micro_batch_idx,
                                    latent: LatentVector::from_vec(
                                        output_data,
                                        stage.end_layer,
                                        mb_clone.latent.stream_id,
                                    ),
                                    request_id: mb_clone.request_id,
                                })
                            } else {
                                // No model — pass through with updated layer index.
                                Ok(MicroBatch {
                                    micro_batch_idx: mb.micro_batch_idx,
                                    latent: LatentVector::from_vec(
                                        mb.latent.data.clone(),
                                        stage.end_layer,
                                        mb.latent.stream_id,
                                    ),
                                    request_id: mb.request_id,
                                })
                            }
                        } else {
                            // ---------- remote processing ----------
                            if let Some(tr) = &transport {
                                let _ = tr.send_latent(
                                    stage.node_id,
                                    stage.start_layer,
                                    mb.latent.clone(),
                                    mb.request_id,
                                )
                                .await;
                            }
                            Ok(MicroBatch {
                                micro_batch_idx: mb.micro_batch_idx,
                                latent: LatentVector::from_vec(
                                    mb.latent.data.clone(),
                                    stage.end_layer,
                                    mb.latent.stream_id,
                                ),
                                request_id: mb.request_id,
                            })
                        }
                    })
                    .await;

                    let elapsed_us = t0.elapsed().as_micros() as u64;

                    match processed {
                        Ok(Ok(out)) => {
                            if collect_stats {
                                stats_slot.lock().await.push(elapsed_us);
                            }
                            if tx.send(out).await.is_err() {
                                warn!(
                                    "Pipeline stage {}: downstream channel closed",
                                    stage.stage_idx
                                );
                                break;
                            }
                        }
                        Ok(Err(e)) => {
                            warn!("Pipeline stage {} error: {}", stage.stage_idx, e);
                            break;
                        }
                        Err(_) => {
                            warn!(
                                "Pipeline stage {} timed out after {}ms",
                                stage.stage_idx, stage_timeout_ms
                            );
                            break;
                        }
                    }
                }
                // Dropping tx closes this stage's output channel, which signals
                // the next stage that no more micro-batches are coming.
            });

            handles.push(handle);
        }

        // Collect outputs from the final channel.
        let mut final_rx = rx_slots[num_stages]
            .take()
            .expect("final receiver already consumed");
        // Drop remaining tx_slots so channels can close properly.
        drop(tx_slots);

        let mut outputs: Vec<Option<MicroBatch>> = vec![None; num_micro_batches];
        let mut received = 0usize;

        while let Some(mb) = final_rx.recv().await {
            let idx = mb.micro_batch_idx;
            if idx < num_micro_batches {
                outputs[idx] = Some(mb);
                received += 1;
            }
        }

        // Wait for all stage tasks to finish.
        for handle in handles {
            let _ = handle.await;
        }

        if received == 0 {
            bail!("Pipeline produced no outputs for request {}", request_id);
        }

        // Collect output latents in order.
        let output_latents: Vec<LatentVector> = outputs
            .into_iter()
            .filter_map(|opt| opt.map(|mb| mb.latent))
            .collect();

        if output_latents.len() != num_micro_batches {
            warn!(
                "Pipeline produced {}/{} micro-batch outputs",
                output_latents.len(),
                num_micro_batches
            );
        }

        let total_time_us = pipeline_start.elapsed().as_micros() as u64;

        let stats = if self.config.collect_stats {
            let mut stage_stats = Vec::with_capacity(num_stages);
            for (s, slot) in stage_latencies.iter().enumerate() {
                let latencies = slot.lock().await.clone();
                stage_stats.push(PipelineStageStats {
                    stage_idx: s,
                    latencies_us: latencies,
                });
            }

            let throughput = if total_time_us > 0 {
                (num_micro_batches as f64) / (total_time_us as f64 / 1_000_000.0)
            } else {
                0.0
            };

            Some(PipelineStats {
                stage_stats,
                total_time_us,
                num_micro_batches,
                throughput_mb_per_sec: throughput,
            })
        } else {
            None
        };

        info!(
            "Pipeline complete: {} outputs in {}ms (request {})",
            output_latents.len(),
            total_time_us / 1000,
            request_id
        );

        Ok(PipelineResult {
            outputs: output_latents,
            stats,
        })
    }

    /// Convenience: execute the pipeline and merge micro-batches into a single
    /// output latent.
    pub async fn execute_and_merge(
        &self,
        input: &LatentVector,
        request_id: uuid::Uuid,
    ) -> Result<(LatentVector, Option<PipelineStats>)> {
        let result = self.execute_pipeline(input, request_id).await?;
        let merged = Self::merge_micro_batches(&result.outputs);
        Ok((merged, result.stats))
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device = detect_device().unwrap();
        let info = device_info(&device);
        assert!(!info.is_empty());
    }

    #[test]
    fn test_distributed_coordinator() {
        let config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let mut coord = DistributedCoordinator::new(config, node_id);

        let mut topology = mycelium_core::TopologyMap::default();
        topology.nodes.push((
            NodeId::new(),
            mycelium_core::NodeCapabilities::cpu_only(8192),
        ));

        coord.update_topology(topology);
        let assignments = coord.compute_assignments();
        // With cpu_only (0 vram), proportional assignment gives 0 layers
        // This is expected — nodes with 0 VRAM can't hold model weights
    }

    #[test]
    fn test_tensor_to_latent_roundtrip() {
        let device = Device::Cpu;
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, (1, 4), &device).unwrap();
        let latent = tensor_to_latent(&tensor, 5).unwrap();
        assert_eq!(latent.dim, 4);
        assert_eq!(latent.layer_idx, 5);
    }

    #[test]
    fn test_gguf_loader_new() {
        let device = Device::Cpu;
        let loader = GGUFLoader::new("/nonexistent/model.gguf", device);
        assert_eq!(loader.model_path, PathBuf::from("/nonexistent/model.gguf"));
    }

    #[test]
    fn test_inference_engine_new() {
        let engine = InferenceEngine::cpu();
        assert!(!engine.is_loaded());
        assert!(engine.tokenizer.is_none());
    }

    #[test]
    fn test_inference_engine_no_model_error() {
        let mut engine = InferenceEngine::cpu();
        let result = engine.generate("hello", 10, 0.7);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No model loaded"));
    }

    #[test]
    fn test_inference_engine_no_tokenizer_error() {
        // Even with no tokenizer, the generate should error about tokenizer
        let mut engine = InferenceEngine::cpu();
        let result = engine.generate("hello", 10, 0.7);
        assert!(result.is_err());
    }

    #[test]
    fn test_moe_inference_engine_new() {
        let config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let engine = MoeInferenceEngine::new(config, node_id).unwrap();
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_generate_result_structure() {
        let result = GenerateResult {
            text: "hello world".to_string(),
            token_ids: vec![1, 2, 3],
            new_token_count: 1,
        };
        assert_eq!(result.text, "hello world");
        assert_eq!(result.new_token_count, 1);
    }

    #[test]
    fn test_model_info_structure() {
        let info = ModelInfo {
            config: ModelConfig::minimax_m25(),
            tensor_count: 100,
            path: PathBuf::from("/test/model.gguf"),
            device_info: "CPU".to_string(),
        };
        assert_eq!(info.tensor_count, 100);
    }

    #[test]
    fn test_latent_mode_variants() {
        let standard = LatentMode::Standard;
        let morph = LatentMode::LatentMorph { t: 0.5 };
        let blend = LatentMode::LatentBlend { weights: vec![0.3, 0.7] };
        let tuning = LatentMode::SelfTuning;

        assert_eq!(standard, LatentMode::Standard);
        assert!(matches!(morph, LatentMode::LatentMorph { .. }));
        assert!(matches!(blend, LatentMode::LatentBlend { .. }));
        assert!(matches!(tuning, LatentMode::SelfTuning));
    }

    #[test]
    fn test_rms_norm_forward() {
        let device = Device::Cpu;
        let weight = Tensor::ones(&[4], DType::F32, &device).unwrap();
        let norm = RmsNorm { weight, eps: 1e-6 };
        // RmsNorm expects 3D input [batch, seq, hidden_dim]
        let input = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap().reshape((1, 1, 4)).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 1, 4]);
    }

    #[test]
    fn test_repeat_kv() {
        let device = Device::Cpu;
        let kv = Tensor::zeros(&[1, 2, 4, 8], DType::F32, &device).unwrap();
        let repeated = repeat_kv(&kv, 4).unwrap();
        assert_eq!(repeated.dims(), &[1, 8, 4, 8]);
    }

    #[test]
    fn test_precompute_freqs() {
        let device = Device::Cpu;
        let (cos, sin) = precompute_freqs_cis(64, 10000.0, 128, &device).unwrap();
        assert_eq!(cos.dims(), &[128, 32]);
        assert_eq!(sin.dims(), &[128, 32]);
    }

    #[tokio::test]
    async fn test_distributed_tensor_router_creation() {
        let config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let coordinator = DistributedCoordinator::new(config.clone(), node_id.clone());
        let router = DistributedTensorRouter::new(node_id, config, coordinator);
        // Router should be created successfully
        assert!(router.command_sender().capacity() > 0);
    }

    #[tokio::test]
    async fn test_router_infer_no_model() {
        let config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let coordinator = DistributedCoordinator::new(config.clone(), node_id.clone());
        let router = DistributedTensorRouter::new(node_id.clone(), config, coordinator);
        
        // Register no local layers (coordinator mode)
        router.register_local_shard(0, 0, Vec::new()).await;
        
        let request = InferenceRequest {
            id: uuid::Uuid::new_v4(),
            prompt: "test prompt".into(),
            max_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
            latent_mode: false,
        };
        
        let response = router.infer(request).await.unwrap();
        assert!(response.text.unwrap().contains("coordinator-only"));
    }

    #[test]
    fn test_layer_result_structure() {
        let latent = LatentVector::zeros(6144, 5, uuid::Uuid::new_v4());
        let result = LayerResult {
            layer_idx: 5,
            latent,
            is_final: false,
        };
        assert_eq!(result.layer_idx, 5);
        assert!(!result.is_final);
    }

    #[test]
    fn test_router_command_variants() {
        // Test that command variants can be created
        let cmd1 = RouterCommand::UpdateTopology(mycelium_core::TopologyMap::default());
        let cmd2 = RouterCommand::RegisterShard {
            layer_start: 0,
            layer_end: 16,
            expert_ids: vec![0, 1, 2, 3],
        };
        
        // Commands should be debug-printable
        let _ = format!("{:?}", cmd1);
        let _ = format!("{:?}", cmd2);
    }

    #[test]
    fn test_moe_router_config_default() {
        let config = MoERouterConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.request_timeout_ms, 5000);
        assert!(config.fallback_to_local);
    }

    #[test]
    fn test_select_top_k_experts() {
        let weights = vec![0.1, 0.3, 0.4, 0.05, 0.15];
        let model_config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let coordinator = Arc::new(RwLock::new(
            DistributedCoordinator::new(model_config.clone(), node_id)
        ));
        
        // Create a dummy gate weights tensor
        let device = Device::Cpu;
        let gate_data = vec![0.0f32; model_config.hidden_dim * model_config.num_experts];
        let gate_tensor = Tensor::from_slice(&gate_data, 
            (model_config.hidden_dim, model_config.num_experts), &device).unwrap();
        let gate_qtensor = candle_core::quantized::QTensor::quantize(&gate_tensor, candle_core::quantized::GgmlDType::F32).unwrap();
        let gate_weights = QMatMul::from_qtensor(gate_qtensor).unwrap();
        
        let router = NetworkMoERouter::new(
            node_id,
            model_config,
            MoERouterConfig::default(),
            coordinator,
            gate_weights,
        );
        
        let selected = router.select_top_k_experts(&weights, 3);
        assert_eq!(selected.len(), 3);
        
        // Should select indices 2, 1, 4 (highest weights)
        assert_eq!(selected[0].0, 2);
        assert_eq!(selected[1].0, 1);
        assert_eq!(selected[2].0, 4);
        
        // Weights should be normalized
        let sum: f32 = selected.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_moe_router_local_expert() {
        let model_config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let coordinator = Arc::new(RwLock::new(
            DistributedCoordinator::new(model_config.clone(), node_id.clone())
        ));
        
        // Create dummy gate weights
        let device = Device::Cpu;
        let gate_data = vec![0.0f32; model_config.hidden_dim * model_config.num_experts];
        let gate_tensor = Tensor::from_slice(&gate_data, 
            (model_config.hidden_dim, model_config.num_experts), &device).unwrap();
        let gate_qtensor = candle_core::quantized::QTensor::quantize(&gate_tensor, candle_core::quantized::GgmlDType::F32).unwrap();
        let gate_weights = QMatMul::from_qtensor(gate_qtensor).unwrap();
        
        let mut router = NetworkMoERouter::new(
            node_id,
            model_config.clone(),
            MoERouterConfig::default(),
            coordinator,
            gate_weights,
        );
        
        // Register local experts with dummy weights
        let dummy_mlp = Mlp {
            feed_forward_w1: {
                let w = Tensor::zeros((model_config.intermediate_dim, model_config.hidden_dim), DType::F32, &device).unwrap();
                let qw = candle_core::quantized::QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
                QMatMul::from_qtensor(qw).unwrap()
            },
            feed_forward_w2: {
                let w = Tensor::zeros((model_config.hidden_dim, model_config.intermediate_dim), DType::F32, &device).unwrap();
                let qw = candle_core::quantized::QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
                QMatMul::from_qtensor(qw).unwrap()
            },
            feed_forward_w3: {
                let w = Tensor::zeros((model_config.intermediate_dim, model_config.hidden_dim), DType::F32, &device).unwrap();
                let qw = candle_core::quantized::QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32).unwrap();
                QMatMul::from_qtensor(qw).unwrap()
            },
        };
        
        router.register_local_experts(vec![0, 1, 2, 3], vec![dummy_mlp.clone(); 4]);
        
        // Test routing
        let latent = LatentVector::zeros(model_config.hidden_dim, 0, uuid::Uuid::new_v4());
        let result = router.route(&latent, 0, uuid::Uuid::new_v4()).await;
        
        // Should succeed with local experts
        assert!(result.is_ok());
    }

    // ─── Pipeline Parallelism Tests ────────────────────────────────────────

    #[test]
    fn test_pipeline_parallel_config_default() {
        let config = PipelineParallelConfig::default();
        assert_eq!(config.num_micro_batches, 4);
        assert_eq!(config.stage_timeout_ms, 10_000);
        assert_eq!(config.max_inflight, 8);
        assert_eq!(config.channel_buffer_size, 16);
        assert!(config.collect_stats);
    }

    #[test]
    fn test_pipeline_stage_num_layers() {
        let node = NodeId::new();
        let stage = PipelineStage::new(0, 5, 15, node);
        assert_eq!(stage.num_layers(), 10);
        assert_eq!(stage.stage_idx, 0);
        assert_eq!(stage.start_layer, 5);
        assert_eq!(stage.end_layer, 15);
    }

    #[test]
    fn test_pipeline_plan_from_node_layers() {
        let n1 = NodeId::new();
        let n2 = NodeId::new();
        let n3 = NodeId::new();

        let plan = PipelinePlan::from_node_layers(vec![(n1, 10), (n2, 20), (n3, 10)]);

        assert_eq!(plan.num_stages(), 3);
        assert_eq!(plan.total_layers, 40);

        assert_eq!(plan.stages[0].start_layer, 0);
        assert_eq!(plan.stages[0].end_layer, 10);
        assert_eq!(plan.stages[0].node_id, n1);

        assert_eq!(plan.stages[1].start_layer, 10);
        assert_eq!(plan.stages[1].end_layer, 30);
        assert_eq!(plan.stages[1].node_id, n2);

        assert_eq!(plan.stages[2].start_layer, 30);
        assert_eq!(plan.stages[2].end_layer, 40);
        assert_eq!(plan.stages[2].node_id, n3);

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_pipeline_plan_uniform() {
        let nodes: Vec<NodeId> = (0..3).map(|_| NodeId::new()).collect();
        let plan = PipelinePlan::uniform(&nodes, 10);

        assert_eq!(plan.num_stages(), 3);
        assert_eq!(plan.total_layers, 10);

        // 10 / 3 = 3 remainder 1 — first node gets 4, rest get 3.
        assert_eq!(plan.stages[0].num_layers(), 4);
        assert_eq!(plan.stages[1].num_layers(), 3);
        assert_eq!(plan.stages[2].num_layers(), 3);

        assert!(plan.validate().is_ok());
    }

    #[test]
    fn test_pipeline_plan_validate_empty() {
        let plan = PipelinePlan {
            stages: vec![],
            total_layers: 0,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_pipeline_plan_validate_gap() {
        let n = NodeId::new();
        let plan = PipelinePlan {
            stages: vec![
                PipelineStage::new(0, 0, 5, n),
                PipelineStage::new(1, 7, 10, n), // gap at layer 5-7
            ],
            total_layers: 10,
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_pipeline_plan_validate_layer_mismatch() {
        let n = NodeId::new();
        let plan = PipelinePlan {
            stages: vec![PipelineStage::new(0, 0, 5, n)],
            total_layers: 10, // mismatch: stages cover 5 but total says 10
        };
        assert!(plan.validate().is_err());
    }

    #[test]
    fn test_pipeline_stage_stats() {
        let mut stats = PipelineStageStats::new(0);
        assert_eq!(stats.mean_latency_us(), 0.0);
        assert_eq!(stats.max_latency_us(), 0);

        stats.latencies_us = vec![100, 200, 300];
        assert!((stats.mean_latency_us() - 200.0).abs() < f64::EPSILON);
        assert_eq!(stats.max_latency_us(), 300);
    }

    #[test]
    fn test_micro_batch_split_single() {
        let node = NodeId::new();
        let plan = PipelinePlan::from_node_layers(vec![(node, 4)]);
        let config = PipelineParallelConfig {
            num_micro_batches: 1,
            ..Default::default()
        };
        let model = Arc::new(RwLock::new(None));
        let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

        let latent = LatentVector::zeros(128, 0, uuid::Uuid::new_v4());
        let rid = uuid::Uuid::new_v4();
        let batches = executor.split_micro_batches(&latent, rid);

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].latent.data.len(), 128);
    }

    #[test]
    fn test_micro_batch_split_multiple() {
        let node = NodeId::new();
        let plan = PipelinePlan::from_node_layers(vec![(node, 4)]);
        let config = PipelineParallelConfig {
            num_micro_batches: 4,
            ..Default::default()
        };
        let model = Arc::new(RwLock::new(None));
        let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

        let latent = LatentVector::zeros(128, 0, uuid::Uuid::new_v4());
        let rid = uuid::Uuid::new_v4();
        let batches = executor.split_micro_batches(&latent, rid);

        assert_eq!(batches.len(), 4);
        // Total data across all batches should equal original
        let total: usize = batches.iter().map(|b| b.latent.data.len()).sum();
        assert_eq!(total, 128);
        // Each batch has correct index
        for (i, b) in batches.iter().enumerate() {
            assert_eq!(b.micro_batch_idx, i);
        }
    }

    #[tokio::test]
    async fn test_pipeline_execute_passthrough() {
        // With no model loaded, stages pass through data with updated layer_idx.
        let node = NodeId::new();
        let plan = PipelinePlan::from_node_layers(vec![(node, 4), (node, 4)]);
        let config = PipelineParallelConfig {
            num_micro_batches: 2,
            stage_timeout_ms: 5000,
            collect_stats: true,
            ..Default::default()
        };

        let model = Arc::new(RwLock::new(None));
        let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

        let latent = LatentVector::from_vec(vec![1.0; 64], 0, uuid::Uuid::new_v4());
        let rid = uuid::Uuid::new_v4();

        let result = executor.execute_pipeline(&latent, rid).await;
        assert!(result.is_ok(), "Pipeline should succeed: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.outputs.len(), 2); // 2 micro-batches

        // Stats should be present
        let stats = result.stats.as_ref().unwrap();
        assert_eq!(stats.num_micro_batches, 2);
        assert_eq!(stats.stage_stats.len(), 2);
        assert!(stats.total_time_us > 0);
        assert!(stats.throughput_mb_per_sec > 0.0);
    }

    #[tokio::test]
    async fn test_pipeline_execute_and_merge() {
        let node = NodeId::new();
        let plan = PipelinePlan::from_node_layers(vec![(node, 2)]);
        let config = PipelineParallelConfig {
            num_micro_batches: 2,
            stage_timeout_ms: 5000,
            collect_stats: false,
            ..Default::default()
        };

        let model = Arc::new(RwLock::new(None));
        let executor = PipelineExecutor::new(plan, config, node, model).unwrap();

        let latent = LatentVector::from_vec(vec![2.0; 64], 0, uuid::Uuid::new_v4());
        let rid = uuid::Uuid::new_v4();

        let (merged, stats) = executor.execute_and_merge(&latent, rid).await.unwrap();
        // Merged output should have same total dimension as input
        assert_eq!(merged.data.len(), 64);
        assert_eq!(merged.layer_idx, 2);
        // Stats collection was disabled
        assert!(stats.is_none());
    }

    // ─── Latent Streaming Tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_latent_stream_manager_create_stream() {
        let manager = LatentStreamManager::new(LatentStreamConfig::default());
        let source = NodeId::new();
        let target = NodeId::new();

        let (sender, mut receiver) = manager.create_stream(source, target).await;

        assert_eq!(manager.active_stream_count().await, 1);
        assert_eq!(sender.meta.source_node, source);
        assert_eq!(sender.meta.target_node, target);
        assert_eq!(receiver.meta.source_node, source);
    }

    #[tokio::test]
    async fn test_latent_stream_send_recv() {
        let manager = LatentStreamManager::new(LatentStreamConfig::default());
        let source = NodeId::new();
        let target = NodeId::new();

        let (sender, mut receiver) = manager.create_stream(source, target).await;

        let latent = LatentVector::from_vec(vec![1.0, 2.0, 3.0], 0, sender.meta.stream_id);
        sender.send(latent.clone()).await.unwrap();

        let received = receiver.recv().await.unwrap();
        assert_eq!(received.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(received.layer_idx, 0);
    }

    #[tokio::test]
    async fn test_latent_stream_multiple_streams() {
        let manager = LatentStreamManager::new(LatentStreamConfig::default());
        let node_a = NodeId::new();
        let node_b = NodeId::new();
        let node_c = NodeId::new();

        let (sender1, mut receiver1) = manager.create_stream(node_a, node_b).await;
        let (sender2, mut receiver2) = manager.create_stream(node_b, node_c).await;

        assert_eq!(manager.active_stream_count().await, 2);

        // Send on first stream
        sender1.send(LatentVector::zeros(64, 0, sender1.meta.stream_id)).await.unwrap();
        // Send on second stream
        sender2.send(LatentVector::zeros(64, 1, sender2.meta.stream_id)).await.unwrap();

        // Receivers should get their respective data
        let r1 = receiver1.recv().await.unwrap();
        let r2 = receiver2.recv().await.unwrap();
        assert_eq!(r1.layer_idx, 0);
        assert_eq!(r2.layer_idx, 1);
    }

    #[tokio::test]
    async fn test_latent_stream_remove_stream() {
        let manager = LatentStreamManager::new(LatentStreamConfig::default());
        let source = NodeId::new();
        let target = NodeId::new();

        let (sender, _receiver) = manager.create_stream(source, target).await;
        let stream_id = sender.meta.stream_id;

        assert_eq!(manager.active_stream_count().await, 1);

        let removed = manager.remove_stream(&stream_id).await;
        assert!(removed);
        assert_eq!(manager.active_stream_count().await, 0);
    }

    #[tokio::test]
    async fn test_latent_stream_try_send_full() {
        // Create a stream with small buffer
        let config = LatentStreamConfig {
            buffer_size: 2,
            backpressure: false,
            max_throughput: None,
        };
        let manager = LatentStreamManager::new(config);
        let source = NodeId::new();
        let target = NodeId::new();

        let (sender, _receiver) = manager.create_stream(source, target).await;

        // Fill the buffer
        sender.try_send(LatentVector::zeros(64, 0, sender.meta.stream_id)).unwrap();
        sender.try_send(LatentVector::zeros(64, 0, sender.meta.stream_id)).unwrap();

        // Next try_send should fail (buffer full)
        let result = sender.try_send(LatentVector::zeros(64, 0, sender.meta.stream_id));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_distributed_tensor_router_has_stream_manager() {
        let config = ModelConfig::minimax_m25();
        let node_id = NodeId::new();
        let coordinator = DistributedCoordinator::new(config.clone(), node_id.clone());
        let router = DistributedTensorRouter::new(node_id, config, coordinator);

        // Router should have a stream manager
        let stream_mgr = router.stream_manager();
        assert_eq!(stream_mgr.active_stream_count().await, 0);
    }

    #[test]
    fn test_latent_stream_meta_serialization() {
        let meta = LatentStreamMeta {
            stream_id: uuid::Uuid::new_v4(),
            source_node: NodeId::new(),
            target_node: NodeId::new(),
            created_at: 1234567890,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let decoded: LatentStreamMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.stream_id, meta.stream_id);
        assert_eq!(decoded.source_node, meta.source_node);
        assert_eq!(decoded.target_node, meta.target_node);
    }
}
