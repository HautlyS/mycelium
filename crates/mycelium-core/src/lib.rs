//! # Mycelium Core — Shared types, constants, and protocols
//!
//! The foundational types that all mycelium crates build upon.
//! Every struct here represents a concept in the mycelium network.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use uuid::Uuid;

// ─── Node Identity ───────────────────────────────────────────────────────

/// Unique identifier for a node in the mycelium network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub Uuid);

impl NodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ─── Hardware Capabilities ───────────────────────────────────────────────

/// What a node can offer to the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// GPU type (e.g., "NVIDIA RTX 4090", "Apple M4", "WebGPU")
    pub gpu_type: GpuType,
    /// Available VRAM in MB
    pub vram_mb: u32,
    /// Available system RAM in MB
    pub ram_mb: u32,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Number of compute units (SMs for CUDA, cores for Metal)
    pub compute_units: u32,
    /// Network bandwidth estimate in Mbps
    pub bandwidth_mbps: u32,
    /// Can this node run in browser? (WASM target)
    pub is_browser: bool,
    /// Can this node store model shards?
    pub can_store: bool,
    /// Can this node run expert layers?
    pub can_compute: bool,
}

impl NodeCapabilities {
    /// Auto-detect the hardware capabilities of this machine.
    ///
    /// Probes system RAM, attempts to detect GPU type and VRAM,
    /// and sets reasonable defaults for the remaining fields.
    pub fn auto_detect() -> Self {
        let ram_mb = Self::detect_ram_mb();
        let (gpu_type, vram_mb, compute_units) = Self::detect_gpu();
        let cpu_cores = Self::detect_cpu_cores();

        Self {
            gpu_type,
            vram_mb,
            ram_mb,
            cpu_cores,
            compute_units,
            bandwidth_mbps: 100, // conservative default
            is_browser: false,
            can_store: ram_mb > 4096,
            can_compute: vram_mb > 0 || ram_mb > 8192,
        }
    }

    /// Detect CPU core count.
    fn detect_cpu_cores() -> u32 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .map(|content| content.matches("processor").count() as u32)
                .unwrap_or(1)
        }
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            Command::new("sysctl")
                .args(["-n", "hw.ncpu"])
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<u32>()
                        .ok()
                })
                .unwrap_or(1)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            1
        }
    }

    /// Detect system RAM in MB.
    fn detect_ram_mb() -> u32 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|line| {
                            line.split_whitespace()
                                .nth(1)
                                .and_then(|v| v.parse::<u64>().ok())
                        })
                })
                .map(|kb| (kb / 1024) as u32)
                .unwrap_or(8192)
        }
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<u64>()
                        .ok()
                })
                .map(|bytes| (bytes / (1024 * 1024)) as u32)
                .unwrap_or(8192)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            8192
        }
    }

    /// Detect GPU type, VRAM, and compute units.
    fn detect_gpu() -> (GpuType, u32, u32) {
        // Try CUDA first
        #[cfg(feature = "cuda")]
        {
            // GPU detection via candle-core's Device::cuda happens at runtime.
            // Static detection returns defaults; actual VRAM/query happens in mycelium-compute.
            return (
                GpuType::Cuda {
                    name: "CUDA GPU".into(),
                    sm_version: 80,
                },
                0,
                0,
            );
        }

        // Try Metal on macOS
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            let metal_name = Command::new("system_profiler")
                .args(["SPDisplaysDataType"])
                .output()
                .ok()
                .and_then(|o| {
                    let s = String::from_utf8_lossy(&o.stdout);
                    s.lines()
                        .find(|l| l.contains("Chipset Model") || l.contains("Metal"))
                        .map(|l| {
                            l.split(':')
                                .nth(1)
                                .map(|s| s.trim().to_string())
                                .unwrap_or_else(|| "Apple GPU".into())
                        })
                        .unwrap_or_else(|| "Apple GPU".into())
                })
                .unwrap_or_else(|| "Apple GPU".into());

            // On Apple Silicon, RAM is shared with GPU (unified memory)
            let ram_mb = Self::detect_ram_mb();
            return (GpuType::Metal { name: metal_name }, ram_mb, 0);
        }

        // Fallback: CPU only
        (GpuType::CpuOnly, 0, 0)
    }

    /// Create capabilities suitable for a browser (WASM) node.
    pub fn browser() -> Self {
        Self {
            gpu_type: GpuType::WebGPU {
                adapter_name: "browser".into(),
            },
            vram_mb: 0,
            ram_mb: 512,
            cpu_cores: 1,
            compute_units: 0,
            bandwidth_mbps: 10,
            is_browser: true,
            can_store: false,
            can_compute: true,
        }
    }

    /// Create a minimal CPU-only capabilities set.
    pub fn cpu_only(ram_mb: u32) -> Self {
        Self {
            gpu_type: GpuType::CpuOnly,
            vram_mb: 0,
            ram_mb,
            cpu_cores: 1,
            compute_units: 0,
            bandwidth_mbps: 50,
            is_browser: false,
            can_store: ram_mb > 4096,
            can_compute: ram_mb > 8192,
        }
    }
}

/// GPU classification for heterogeneous scheduling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuType {
    /// NVIDIA CUDA GPU
    Cuda { name: String, sm_version: u32 },
    /// Apple Silicon Metal
    Metal { name: String },
    /// WebGPU in browser (WASM)
    WebGPU { adapter_name: String },
    /// CPU only (slow but works)
    CpuOnly,
}

// ─── Latent Space ────────────────────────────────────────────────────────

/// A latent vector — the continuous representation that flows through the
/// mycelium network instead of tokens.
///
/// This is the fundamental unit of information exchange between nodes.
/// Unlike tokens (discrete integers), latent vectors are continuous f32 arrays
/// that can be interpolated, morphed, blended, and processed with matrix ops.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentVector {
    /// The actual data (hidden_dim floats, typically 6144 for MiniMax M2.5)
    pub data: Vec<f32>,
    /// Dimensionality of this latent space
    pub dim: usize,
    /// Which layer produced this latent
    pub layer_idx: usize,
    /// Stream ID for tracking continuous flows
    pub stream_id: Uuid,
}

impl LatentVector {
    /// Create a zero latent vector of given dimension
    pub fn zeros(dim: usize, layer_idx: usize, stream_id: Uuid) -> Self {
        Self {
            data: vec![0.0; dim],
            dim,
            layer_idx,
            stream_id,
        }
    }

    /// Create from raw data
    pub fn from_vec(data: Vec<f32>, layer_idx: usize, stream_id: Uuid) -> Self {
        let dim = data.len();
        Self {
            data,
            dim,
            layer_idx,
            stream_id,
        }
    }

    /// Interpolate between two latent vectors
    pub fn lerp(&self, other: &LatentVector, t: f32) -> LatentVector {
        assert_eq!(self.dim, other.dim, "Latent dimensions must match for lerp");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * (1.0 - t) + b * t)
            .collect();
        LatentVector {
            data,
            dim: self.dim,
            layer_idx: self.layer_idx,
            stream_id: self.stream_id,
        }
    }

    /// Add two latent vectors (residual connection)
    pub fn add(&self, other: &LatentVector) -> LatentVector {
        assert_eq!(self.dim, other.dim);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        LatentVector {
            data,
            dim: self.dim,
            layer_idx: other.layer_idx,
            stream_id: self.stream_id,
        }
    }

    /// Scale by a scalar (for attention weighting)
    pub fn scale(&self, s: f32) -> LatentVector {
        let data = self.data.iter().map(|&x| x * s).collect();
        LatentVector {
            data,
            dim: self.dim,
            layer_idx: self.layer_idx,
            stream_id: self.stream_id,
        }
    }

    /// L2 norm of the latent vector
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> LatentVector {
        let n = self.norm();
        if n > 1e-8 {
            self.scale(1.0 / n)
        } else {
            self.clone()
        }
    }

    /// Cosine similarity with another latent vector
    pub fn cosine_similarity(&self, other: &LatentVector) -> f32 {
        let dot: f32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        dot / (self.norm() * other.norm()).max(1e-8)
    }

    /// Convert to bytes for network transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.data.len() * 4);
        for f in &self.data {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    /// Convert from bytes
    pub fn from_bytes(data: &[u8], dim: usize, layer_idx: usize, stream_id: Uuid) -> Self {
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self {
            data: floats,
            dim,
            layer_idx,
            stream_id,
        }
    }
}

// ─── Model Architecture ──────────────────────────────────────────────────

/// MiniMax M2.5 MoE architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,
    /// Hidden dimension (6144 for M2.5)
    pub hidden_dim: usize,
    /// Number of transformer layers (64 for M2.5)
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of experts in MoE (64 for M2.5)
    pub num_experts: usize,
    /// Number of active experts per token (4 for M2.5)
    pub top_k_experts: usize,
    /// Intermediate dimension in FFN
    pub intermediate_dim: usize,
    /// Maximum context length
    pub max_context: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// EOS token ID (for stopping generation)
    pub eos_token_id: Option<TokenId>,
    /// BOS token ID
    pub bos_token_id: Option<TokenId>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::minimax_m25()
    }
}

impl ModelConfig {
    /// MiniMax M2.5 230B MoE configuration
    pub fn minimax_m25() -> Self {
        Self {
            name: "MiniMax-M2.5".into(),
            hidden_dim: 6144,
            num_layers: 64,
            num_heads: 48,
            head_dim: 128,
            num_experts: 64,
            top_k_experts: 4,
            intermediate_dim: 16384,
            max_context: 1_000_000,
            vocab_size: 200000,
            eos_token_id: Some(2), // Common EOS token ID
            bos_token_id: Some(1), // Common BOS token ID
        }
    }

    /// Estimate VRAM needed for this model at given quantization
    pub fn vram_estimate_mb(&self, quant_bits: u8) -> u32 {
        let bytes_per_param = quant_bits as f64 / 8.0;
        let total_bytes = 456_000_000_000.0 * bytes_per_param;
        (total_bytes / (1024.0 * 1024.0)) as u32
    }

    /// How many layers can fit in given VRAM?
    pub fn layers_for_vram(&self, vram_mb: u32, quant_bits: u8) -> usize {
        let bytes_per_param = quant_bits as f64 / 8.0;
        let params_per_layer = 456_000_000_000.0 / self.num_layers as f64;
        let mb_per_layer = (params_per_layer * bytes_per_param) / (1024.0 * 1024.0);
        (vram_mb as f64 / mb_per_layer).floor() as usize
    }
}

// ─── GGUF Config ─────────────────────────────────────────────────────────

/// Configuration parsed from GGUF file metadata.
///
/// GGUF files carry key-value metadata that describes the model architecture.
/// This struct extracts the relevant fields needed to configure inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufConfig {
    /// Model architecture (e.g., "llama", "minimax")
    pub architecture: String,
    /// Model name from metadata
    pub name: String,
    /// Context length the model was trained with
    pub context_length: usize,
    /// Embedding dimension
    pub embedding_length: usize,
    /// Number of layers / blocks
    pub block_count: usize,
    /// Number of attention heads
    pub head_count: usize,
    /// Number of KV heads (may differ from head_count for GQA)
    pub head_count_kv: usize,
    /// Dimension per attention head
    pub feed_forward_length: usize,
    /// Number of experts (1 = dense, >1 = MoE)
    pub expert_count: usize,
    /// Number of active experts per token
    pub expert_used_count: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Quantization version string
    pub quantization_version: Option<String>,
    /// File type string (e.g., "Q4_K_M")
    pub file_type: Option<String>,
    /// All raw GGUF key-value metadata (for extensibility)
    pub metadata: HashMap<String, GgufValue>,
}

/// A value from GGUF metadata. GGUF supports string, uint, float, and array types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value")]
pub enum GgufValue {
    String(String),
    Uint(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<GgufValue>),
}

impl GgufConfig {
    /// Build a GgufConfig from raw GGUF key-value metadata.
    ///
    /// Standard GGUF keys are mapped to struct fields; unrecognized keys
    /// are stored in the `metadata` hashmap for later use.
    pub fn from_metadata(kv: HashMap<String, GgufValue>) -> Self {
        let get_str = |key: &str| -> String {
            kv.get(key)
                .and_then(|v| match v {
                    GgufValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        };
        let get_uint = |key: &str, default: usize| -> usize {
            kv.get(key)
                .and_then(|v| match v {
                    GgufValue::Uint(n) => Some(*n as usize),
                    GgufValue::Int(n) => Some(*n as usize),
                    _ => None,
                })
                .unwrap_or(default)
        };
        let get_float = |key: &str, default: f32| -> f32 {
            kv.get(key)
                .and_then(|v| match v {
                    GgufValue::Float(f) => Some(*f as f32),
                    GgufValue::Int(n) => Some(*n as f32),
                    _ => None,
                })
                .unwrap_or(default)
        };

        Self {
            architecture: get_str("general.architecture"),
            name: get_str("general.name"),
            context_length: get_uint("llama.context_length", 4096),
            embedding_length: get_uint("llama.embedding_length", 4096),
            block_count: get_uint("llama.block_count", 32),
            head_count: get_uint("llama.attention.head_count", 32),
            head_count_kv: get_uint(
                "llama.attention.head_count_kv",
                get_uint("llama.attention.head_count", 32),
            ),
            feed_forward_length: get_uint("llama.feed_forward_length", 11008),
            expert_count: get_uint("llama.expert_count", 1),
            expert_used_count: get_uint("llama.expert_used_count", 1),
            layer_norm_eps: get_float("llama.attention.layer_norm_rms_epsilon", 1e-6),
            vocab_size: get_uint("llama.vocab_size", 32000),
            quantization_version: kv
                .get("general.quantization_version")
                .and_then(|v| match v {
                    GgufValue::String(s) => Some(s.clone()),
                    GgufValue::Uint(n) => Some(format!("{}", n)),
                    _ => None,
                }),
            file_type: kv.get("general.file_type").and_then(|v| match v {
                GgufValue::String(s) => Some(s.clone()),
                GgufValue::Uint(n) => Some(format!("{}", n)),
                _ => None,
            }),
            metadata: kv,
        }
    }

    /// Convert to a ModelConfig, mapping GGUF fields to mycelium's model type.
    ///
    /// For MoE models (expert_count > 1), the expert fields are populated.
    /// For dense models, num_experts = 1 and top_k_experts = 1.
    pub fn to_model_config(&self) -> ModelConfig {
        let head_dim = if self.head_count > 0 {
            self.embedding_length / self.head_count
        } else {
            128
        };
        ModelConfig {
            name: self.name.clone(),
            hidden_dim: self.embedding_length,
            num_layers: self.block_count,
            num_heads: self.head_count,
            head_dim,
            num_experts: self.expert_count.max(1),
            top_k_experts: self.expert_used_count.max(1).min(self.expert_count.max(1)),
            intermediate_dim: self.feed_forward_length,
            max_context: self.context_length,
            vocab_size: self.vocab_size,
            eos_token_id: None,
            bos_token_id: None,
        }
    }

    /// Head dimension (embedding_length / head_count).
    pub fn head_dim(&self) -> usize {
        if self.head_count > 0 {
            self.embedding_length / self.head_count
        } else {
            128
        }
    }

    /// Whether this is a MoE model.
    pub fn is_moe(&self) -> bool {
        self.expert_count > 1
    }
}

// ─── Layer Weights ───────────────────────────────────────────────────────

/// Weight tensors for a single transformer layer.
///
/// Holds references to the weight data for one layer of the model,
/// as loaded from a GGUF shard. The actual storage is a byte buffer
/// (zero-copy mapped from the file when possible); the `offset` and
/// `shape` fields describe how to interpret each tensor.
///
/// This type is Clone + Send + Sync so it can be shared across
/// inference tasks on different threads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Which layer index this represents
    pub layer_idx: usize,
    /// Attention Q weight
    pub attn_q: TensorMeta,
    /// Attention K weight
    pub attn_k: TensorMeta,
    /// Attention V weight
    pub attn_v: TensorMeta,
    /// Attention output projection weight
    pub attn_output: TensorMeta,
    /// Pre-attention layer norm weight
    pub attn_norm: TensorMeta,
    /// FFN gate weight (first of SwiGLU pair)
    pub ffn_gate: TensorMeta,
    /// FFN up weight (second of SwiGLU pair)
    pub ffn_up: TensorMeta,
    /// FFN down projection weight
    pub ffn_down: TensorMeta,
    /// Post-FFN layer norm weight
    pub ffn_norm: TensorMeta,
    /// Expert gate weights (MoE only; empty for dense layers)
    pub expert_gate: Option<TensorMeta>,
    /// Per-expert FFN weights (MoE only; empty for dense layers)
    pub expert_ffns: Vec<ExpertWeights>,
}

/// Weight tensors for a single MoE expert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertWeights {
    /// Expert index
    pub expert_id: usize,
    /// Gate projection
    pub gate: TensorMeta,
    /// Up projection
    pub up: TensorMeta,
    /// Down projection
    pub down: TensorMeta,
}

/// Metadata describing a weight tensor's layout in the weight buffer.
///
/// This does NOT hold the actual tensor data — only the shape, dtype,
/// and offset into the weight buffer. The weight buffer itself is owned
/// by the model shard (mapped from the GGUF file).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorMeta {
    /// Tensor name in the GGUF file (e.g., "blk.0.attn_q.weight")
    pub name: String,
    /// Shape of the tensor [dim0, dim1, ...]
    pub shape: Vec<usize>,
    /// Data type of the tensor elements
    pub dtype: WeightDtype,
    /// Byte offset into the weight buffer
    pub offset: u64,
    /// Byte length of this tensor's data
    pub byte_len: u64,
}

impl TensorMeta {
    /// Number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes for one element of this dtype.
    pub fn element_size(&self) -> usize {
        self.dtype.size_of()
    }
}

/// Data types used for weight tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightDtype {
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE half)
    F16,
    /// 16-bit brain float
    BF16,
    /// 8-bit integer (quantized)
    Q8,
    /// 4-bit quantized (GGUF block format)
    Q4,
    /// 2-bit quantized (GGUF block format)
    Q2,
    /// Generic quantized with given block size
    Quantized { bits: u8, group_size: usize },
}

impl WeightDtype {
    /// Bytes per element for this dtype.
    pub fn size_of(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::Q8 => 1,
            Self::Q4 => 1, // packed
            Self::Q2 => 1, // packed
            Self::Quantized { bits, group_size } => {
                // Approximate: bits per element, packed into bytes
                #[allow(clippy::manual_div_ceil)]
                let bytes_per_group = ((*bits as usize * *group_size) + 7) / 8;
                bytes_per_group / group_size
            }
        }
    }
}

// ─── KV Cache ────────────────────────────────────────────────────────────

/// Key-Value cache for autoregressive generation.
///
/// During inference, the K and V tensors from attention are cached
/// so that previously-computed positions do not need to be recomputed.
/// This struct holds the cache for all layers.
///
/// The cache is organized as:
///   - One entry per layer
///   - Each entry has K and V tensors of shape [batch, n_kv_heads, seq_len, head_dim]
///
/// For multi-query attention or grouped-query attention, n_kv_heads may
/// be less than num_heads, and the K/V tensors are broadcast during
/// attention computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCache {
    /// Per-layer cache entries
    pub layers: Vec<KVCacheEntry>,
    /// Number of layers this cache covers
    pub num_layers: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Current sequence length (number of positions cached)
    pub seq_len: usize,
    /// Maximum sequence length this cache can hold
    pub max_seq_len: usize,
}

/// Cache entry for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVCacheEntry {
    /// Key cache: flat f32 buffer, logically [n_kv_heads, seq_len, head_dim]
    pub k: Vec<f32>,
    /// Value cache: flat f32 buffer, logically [n_kv_heads, seq_len, head_dim]
    pub v: Vec<f32>,
}

impl KVCache {
    /// Create a new empty KV cache.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let slot_size = num_kv_heads * head_dim;
        Self {
            layers: (0..num_layers)
                .map(|_| KVCacheEntry {
                    k: Vec::with_capacity(slot_size * max_seq_len),
                    v: Vec::with_capacity(slot_size * max_seq_len),
                })
                .collect(),
            num_layers,
            num_kv_heads,
            head_dim,
            seq_len: 0,
            max_seq_len,
        }
    }

    /// Append a single token's K/V to the cache.
    ///
    /// `k` and `v` should each be of length `num_kv_heads * head_dim`.
    pub fn append(&mut self, layer_idx: usize, k: &[f32], v: &[f32]) {
        if layer_idx < self.layers.len() {
            self.layers[layer_idx].k.extend_from_slice(k);
            self.layers[layer_idx].v.extend_from_slice(v);
        }
        if layer_idx == 0 {
            // Only increment seq_len once (on the first layer)
            self.seq_len += 1;
        }
    }

    /// Append a batch of tokens' K/V to the cache.
    ///
    /// `k` and `v` should each be of length `num_kv_heads * num_tokens * head_dim`.
    pub fn append_batch(&mut self, layer_idx: usize, k: &[f32], v: &[f32], num_tokens: usize) {
        if layer_idx < self.layers.len() {
            self.layers[layer_idx].k.extend_from_slice(k);
            self.layers[layer_idx].v.extend_from_slice(v);
        }
        if layer_idx == 0 {
            self.seq_len += num_tokens;
        }
    }

    /// Get the K cache for a layer as a slice.
    pub fn k(&self, layer_idx: usize) -> &[f32] {
        &self.layers[layer_idx].k
    }

    /// Get the V cache for a layer as a slice.
    pub fn v(&self, layer_idx: usize) -> &[f32] {
        &self.layers[layer_idx].v
    }

    /// Clear the entire cache (e.g., for a new sequence).
    pub fn clear(&mut self) {
        for entry in &mut self.layers {
            entry.k.clear();
            entry.v.clear();
        }
        self.seq_len = 0;
    }

    /// Check if the cache is full.
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }

    /// Memory usage of the cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        let per_layer = self.num_kv_heads * self.seq_len * self.head_dim * 4; // 2 (K+V) * 4 bytes/f32
        per_layer * self.num_layers * 2
    }
}

// ─── Tokenizer ───────────────────────────────────────────────────────────

/// Trait for tokenizing text and detokenizing token IDs.
///
/// This abstracts over different tokenizer implementations (BPE, SentencePiece, etc.)
/// so that the core crate does not depend on a specific tokenizer library.
///
/// Implementations are provided in mycelium-compute; this trait defines the
/// interface that all tokenizer backends must satisfy.
pub trait Tokenizer: Send + Sync {
    /// Encode a text string into a sequence of token IDs.
    fn encode(&self, text: &str) -> anyhow::Result<Vec<TokenId>>;

    /// Decode a sequence of token IDs back into text.
    fn decode(&self, tokens: &[TokenId]) -> anyhow::Result<String>;

    /// Decode a single token ID to its string representation.
    fn decode_token(&self, token: TokenId) -> anyhow::Result<String>;

    /// Number of tokens in the vocabulary.
    fn vocab_size(&self) -> usize;

    /// The token ID used for beginning-of-sequence.
    fn bos_token_id(&self) -> Option<TokenId>;

    /// The token ID used for end-of-sequence.
    fn eos_token_id(&self) -> Option<TokenId>;

    /// The token ID used for padding.
    fn pad_token_id(&self) -> Option<TokenId>;

    /// Whether this tokenizer adds a BOS token automatically during encode.
    fn adds_bos_token(&self) -> bool {
        false
    }
}

/// A token ID — an index into the model's vocabulary.
pub type TokenId = u32;

/// A simple byte-level tokenizer for fallback / testing purposes.
#[derive(Debug, Clone)]
pub struct ByteTokenizer {
    pub vocab_size: usize,
    pub bos_id: Option<TokenId>,
    pub eos_id: Option<TokenId>,
    pub pad_id: Option<TokenId>,
}

impl ByteTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            bos_id: None,
            eos_id: None,
            pad_id: None,
        }
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<TokenId>> {
        Ok(text.bytes().map(|b| b as TokenId).collect())
    }

    fn decode(&self, tokens: &[TokenId]) -> anyhow::Result<String> {
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn decode_token(&self, token: TokenId) -> anyhow::Result<String> {
        Ok(String::from_utf8_lossy(&[token as u8]).into_owned())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn bos_token_id(&self) -> Option<TokenId> {
        self.bos_id
    }

    fn eos_token_id(&self) -> Option<TokenId> {
        self.eos_id
    }

    fn pad_token_id(&self) -> Option<TokenId> {
        self.pad_id
    }
}

// ─── Content Hashing ─────────────────────────────────────────────────────

/// Compute the SHA-256 hash of arbitrary data.
pub fn sha256_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ─── Layer Assignment ─────────────────────────────────────────────────────

/// Which node owns which part of the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    /// Node responsible for this layer
    pub node_id: NodeId,
    /// Layer index range [start, end)
    pub layer_start: usize,
    pub layer_end: usize,
    /// Expert IDs this node handles (for MoE layers)
    pub expert_ids: Vec<usize>,
    /// Priority for this assignment (lower = preferred)
    pub priority: u32,
}

// ─── Network Topology ─────────────────────────────────────────────────────

/// The current view of the network topology.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopologyMap {
    /// All known nodes and their capabilities
    pub nodes: Vec<(NodeId, NodeCapabilities)>,
    /// Current layer assignments
    pub assignments: Vec<LayerAssignment>,
    /// Pairwise latency estimates (ms) between nodes
    pub latencies: Vec<(NodeId, NodeId, f32)>,
    /// Pairwise bandwidth estimates (Mbps)
    pub bandwidths: Vec<(NodeId, NodeId, f32)>,
}

impl TopologyMap {
    /// Find the best node to run a given layer.
    pub fn best_node_for_layer(&self, layer_idx: usize) -> Option<NodeId> {
        self.assignments
            .iter()
            .find(|a| layer_idx >= a.layer_start && layer_idx < a.layer_end)
            .map(|a| a.node_id)
    }

    /// Find which nodes can run a given expert.
    pub fn nodes_for_expert(&self, expert_id: usize) -> Vec<NodeId> {
        self.assignments
            .iter()
            .filter(|a| a.expert_ids.contains(&expert_id))
            .map(|a| a.node_id)
            .collect()
    }

    /// Estimate total compute capacity.
    pub fn total_vram_mb(&self) -> u32 {
        self.nodes.iter().map(|(_, cap)| cap.vram_mb).sum()
    }
}

// ─── Messages ─────────────────────────────────────────────────────────────

/// All messages that flow through the hyphae (P2P network).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyphaeMessage {
    // ── Discovery ──
    /// Announce node presence and capabilities
    NodeAnnounce {
        node_id: NodeId,
        capabilities: NodeCapabilities,
        listen_addr: SocketAddr,
    },
    /// Node gracefully departing
    NodeDeparture { node_id: NodeId },

    // ── Compute (Distributed Inference) ──
    /// Send a latent vector to another node for processing
    LatentDispatch {
        stream_id: Uuid,
        layer_idx: usize,
        latent: LatentVector,
    },
    /// Result of processing a latent vector
    LatentResult {
        stream_id: Uuid,
        layer_idx: usize,
        latent: LatentVector,
    },
    /// Request expert processing (MoE routing)
    ExpertRequest {
        stream_id: Uuid,
        layer_idx: usize,
        expert_id: usize,
        latent: LatentVector,
    },
    /// Expert processing result
    ExpertResponse {
        stream_id: Uuid,
        layer_idx: usize,
        expert_id: usize,
        latent: LatentVector,
    },

    // ── Continuous Latent Streaming ──
    /// Open a new latent stream between nodes
    StreamOpen {
        stream_id: Uuid,
        source_node: NodeId,
        target_node: NodeId,
        buffer_size: usize,
        layer_start: usize,
        layer_end: usize,
    },
    /// Stream data message carrying a latent vector in an active stream
    StreamData {
        stream_id: Uuid,
        sequence: u64,
        latent: LatentVector,
    },
    /// Acknowledge receipt of stream data (flow control)
    StreamAck {
        stream_id: Uuid,
        sequence: u64,
        received_count: u64,
    },
    /// Close a latent stream
    StreamClose { stream_id: Uuid, reason: String },

    // ── Replication (Spore Protocol) ──
    /// Broadcast that a spore is available for germination
    SporeAvailable {
        spore_id: Uuid,
        model_name: String,
        shard_count: usize,
        total_size_mb: u32,
    },
    /// Request a spore for germination
    SporeRequest { spore_id: Uuid, requester: NodeId },
    /// Transfer a chunk of spore data
    SporeChunk {
        spore_id: Uuid,
        chunk_idx: usize,
        data: Vec<u8>,
    },

    // ── Self-Tuning (Nucleus) ──
    /// Share a LoRA gradient delta with the network
    GradientDelta {
        layer_idx: usize,
        delta: Vec<f32>,
        version: u64,
        node_id: NodeId,
    },
    /// Request weight synchronization
    WeightSyncRequest { from_version: u64, node_id: NodeId },
    /// Weight sync response
    WeightSyncResponse { version: u64, weights: Vec<f32> },

    // ── Topology ──
    /// Updated topology map
    TopologyUpdate { map: TopologyMap },
}

// ─── Inference Request ────────────────────────────────────────────────────

/// A request to run inference through the distributed model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: Uuid,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Latent mode: return latent vectors instead of tokens
    pub latent_mode: bool,
}

/// A response from the distributed model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub id: Uuid,
    /// Generated text (if not latent_mode)
    pub text: Option<String>,
    /// Latent vectors (if latent_mode, or for self-tuning)
    pub latents: Vec<LatentVector>,
    /// Which nodes participated
    pub participating_nodes: Vec<NodeId>,
    /// Time taken in ms
    pub latency_ms: u64,
}

// ─── Spore ────────────────────────────────────────────────────────────────

/// A spore — the self-replication unit.
/// Contains everything needed to spawn a new node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Spore {
    pub id: Uuid,
    /// Compressed model weights (GGUF shard)
    pub genome: SporeGenome,
    /// Learned LoRA adapter
    pub instincts: Option<LoRAAdapter>,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Which layers this spore covers
    pub layer_range: (usize, usize),
    /// Which experts this spore handles
    pub expert_ids: Vec<usize>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Parent node that created this spore
    pub parent: NodeId,
    /// Generation number (for tracking replication depth)
    pub generation: u32,
}

/// The genome — compressed model weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeGenome {
    /// GGUF shard data (compressed)
    pub data: Vec<u8>,
    /// Quantization level
    pub quant_bits: u8,
    /// SHA256 hash for verification
    pub hash: String,
    /// Size in bytes when decompressed
    pub decompressed_size: u64,
}

impl SporeGenome {
    /// Create a new genome from raw data, computing the SHA256 hash.
    pub fn new(data: Vec<u8>, quant_bits: u8, decompressed_size: u64) -> Self {
        let hash = sha256_hash(&data);
        Self {
            data,
            quant_bits,
            hash,
            decompressed_size,
        }
    }

    /// Verify the integrity of the genome data against its stored hash.
    pub fn verify(&self) -> bool {
        sha256_hash(&self.data) == self.hash
    }
}

/// A LoRA adapter — learned modifications to the base model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAAdapter {
    /// Rank of the LoRA decomposition
    pub rank: usize,
    /// A matrices (down-projection)
    pub a_weights: Vec<Vec<f32>>,
    /// B matrices (up-projection)
    pub b_weights: Vec<Vec<f32>>,
    /// Which layers this adapter modifies
    pub target_layers: Vec<usize>,
    /// Scaling factor
    pub alpha: f32,
}

// ─── Constants ────────────────────────────────────────────────────────────

/// Default P2P listen port
pub const DEFAULT_PORT: u16 = 4219;

/// Protocol name for libp2p identification
pub const PROTOCOL_NAME: &str = "mycelium/0.1.0";

/// Gossipsub topic for spore broadcasts
pub const TOPIC_SPORE: &str = "mycelium/spore";

/// Gossipsub topic for gradient sharing
pub const TOPIC_GRADIENT: &str = "mycelium/gradient";

/// Gossipsub topic for topology updates
pub const TOPIC_TOPOLOGY: &str = "mycelium/topology";

/// Maximum latent vector size for network transmission (6144 f32s = 24KB)
pub const MAX_LATENT_SIZE: usize = 6144;

/// How often to broadcast node announcements (seconds)
pub const ANNOUNCE_INTERVAL_SECS: u64 = 30;

/// How often to sync topology (seconds)
pub const TOPOLOGY_SYNC_INTERVAL_SECS: u64 = 60;

/// LoRA default rank
pub const LORA_DEFAULT_RANK: usize = 16;

/// Federated averaging minimum participants
pub const FEDAVG_MIN_PARTICIPANTS: usize = 3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_lerp() {
        let id = Uuid::new_v4();
        let a = LatentVector::from_vec(vec![1.0, 0.0, 0.0], 0, id);
        let b = LatentVector::from_vec(vec![0.0, 1.0, 0.0], 0, id);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.data[0] - 0.5).abs() < 1e-6);
        assert!((mid.data[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_latent_cosine() {
        let id = Uuid::new_v4();
        let a = LatentVector::from_vec(vec![1.0, 0.0], 0, id);
        let b = LatentVector::from_vec(vec![0.0, 1.0], 0, id);
        assert!(a.cosine_similarity(&b).abs() < 1e-6); // orthogonal
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-6); // identical
    }

    #[test]
    fn test_model_config_vram() {
        let config = ModelConfig::minimax_m25();
        let vram = config.vram_estimate_mb(4);
        assert!(vram > 100_000, "Q4 should need >100GB: got {vram}MB");
    }

    #[test]
    fn test_layers_for_vram() {
        let config = ModelConfig::minimax_m25();
        let layers = config.layers_for_vram(81920, 4);
        assert!(layers > 0, "Should fit at least 1 layer in 80GB");
    }

    #[test]
    fn test_sha256_hash() {
        let data = b"hello mycelium";
        let hash = sha256_hash(data);
        assert!(!hash.is_empty(), "Hash should not be empty");
        // Same input = same hash
        assert_eq!(hash, sha256_hash(data));
        // Different input = different hash
        let hash2 = sha256_hash(b"hello myceliuN");
        assert_ne!(hash, hash2);
    }

    #[test]
    fn test_spore_genome_verify() {
        let genome = SporeGenome::new(vec![1, 2, 3, 4], 4, 1024);
        assert!(genome.verify(), "Freshly created genome should verify");
        // Tamper with data
        let mut bad_genome = genome.clone();
        bad_genome.data[0] = 99;
        assert!(!bad_genome.verify(), "Tampered genome should fail verify");
    }

    #[test]
    fn test_kv_cache_basic() {
        let mut cache = KVCache::new(4, 8, 128, 2048);
        assert_eq!(cache.seq_len, 0);

        // Append a token to layer 0
        let k = vec![0.1f32; 8 * 128];
        let v = vec![0.2f32; 8 * 128];
        cache.append(0, &k, &v);
        assert_eq!(cache.seq_len, 1);

        // Append a token to layer 1
        cache.append(1, &k, &v);
        // seq_len should still be 1 (only incremented on layer 0)
        assert_eq!(cache.seq_len, 1);

        // Clear
        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_kv_cache_batch() {
        let mut cache = KVCache::new(2, 4, 64, 512);
        let k = vec![0.0f32; 4 * 3 * 64]; // 3 tokens
        let v = vec![0.0f32; 4 * 3 * 64];
        cache.append_batch(0, &k, &v, 3);
        cache.append_batch(1, &k, &v, 3);
        assert_eq!(cache.seq_len, 3);
    }

    #[test]
    fn test_byte_tokenizer() {
        let tok = ByteTokenizer::new(256);
        let encoded = tok.encode("hello").unwrap();
        assert_eq!(
            encoded,
            vec![
                b'h' as u32,
                b'e' as u32,
                b'l' as u32,
                b'l' as u32,
                b'o' as u32
            ]
        );
        let decoded = tok.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_gguf_config_from_metadata() {
        let mut kv = HashMap::new();
        kv.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        kv.insert(
            "general.name".into(),
            GgufValue::String("test-model".into()),
        );
        kv.insert("llama.context_length".into(), GgufValue::Uint(8192));
        kv.insert("llama.embedding_length".into(), GgufValue::Uint(4096));
        kv.insert("llama.block_count".into(), GgufValue::Uint(32));
        kv.insert("llama.attention.head_count".into(), GgufValue::Uint(32));
        kv.insert("llama.feed_forward_length".into(), GgufValue::Uint(11008));

        let config = GgufConfig::from_metadata(kv);
        assert_eq!(config.architecture, "llama");
        assert_eq!(config.context_length, 8192);
        assert_eq!(config.embedding_length, 4096);
        assert!(!config.is_moe());

        let model = config.to_model_config();
        assert_eq!(model.hidden_dim, 4096);
        assert_eq!(model.num_layers, 32);
        assert_eq!(model.head_dim, 128);
    }

    #[test]
    fn test_gguf_config_moe() {
        let mut kv = HashMap::new();
        kv.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        kv.insert("general.name".into(), GgufValue::String("moe-test".into()));
        kv.insert("llama.embedding_length".into(), GgufValue::Uint(6144));
        kv.insert("llama.block_count".into(), GgufValue::Uint(64));
        kv.insert("llama.attention.head_count".into(), GgufValue::Uint(48));
        kv.insert("llama.expert_count".into(), GgufValue::Uint(64));
        kv.insert("llama.expert_used_count".into(), GgufValue::Uint(4));

        let config = GgufConfig::from_metadata(kv);
        assert!(config.is_moe());

        let model = config.to_model_config();
        assert_eq!(model.num_experts, 64);
        assert_eq!(model.top_k_experts, 4);
    }

    #[test]
    fn test_tensor_meta() {
        let meta = TensorMeta {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![6144, 6144],
            dtype: WeightDtype::Q4,
            offset: 0,
            byte_len: 18874368,
        };
        assert_eq!(meta.num_elements(), 6144 * 6144);
        assert_eq!(meta.element_size(), 1);
    }

    #[test]
    fn test_weight_dtype_sizes() {
        assert_eq!(WeightDtype::F32.size_of(), 4);
        assert_eq!(WeightDtype::F16.size_of(), 2);
        assert_eq!(WeightDtype::BF16.size_of(), 2);
        assert_eq!(WeightDtype::Q8.size_of(), 1);
    }

    #[test]
    fn test_node_capabilities_auto_detect() {
        let caps = NodeCapabilities::auto_detect();
        // Should always have some RAM
        assert!(caps.ram_mb > 0, "Should detect some RAM");
        // CPU-only nodes should have 0 VRAM
        if caps.gpu_type == GpuType::CpuOnly {
            assert_eq!(caps.vram_mb, 0);
        }
    }

    #[test]
    fn test_node_capabilities_cpu_only() {
        let caps = NodeCapabilities::cpu_only(16384);
        assert_eq!(caps.gpu_type, GpuType::CpuOnly);
        assert_eq!(caps.ram_mb, 16384);
        assert_eq!(caps.vram_mb, 0);
        assert!(caps.can_store);
        assert!(caps.can_compute);
    }

    #[test]
    fn test_node_capabilities_browser() {
        let caps = NodeCapabilities::browser();
        assert!(matches!(caps.gpu_type, GpuType::WebGPU { .. }));
        assert!(caps.is_browser);
    }

    // ─── Stream Message Tests ──────────────────────────────────────────

    #[test]
    fn test_hyphae_message_stream_open() {
        let msg = HyphaeMessage::StreamOpen {
            stream_id: Uuid::new_v4(),
            source_node: NodeId::new(),
            target_node: NodeId::new(),
            buffer_size: 128,
            layer_start: 0,
            layer_end: 32,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: HyphaeMessage = serde_json::from_str(&json).unwrap();

        match decoded {
            HyphaeMessage::StreamOpen {
                buffer_size,
                layer_start,
                layer_end,
                ..
            } => {
                assert_eq!(buffer_size, 128);
                assert_eq!(layer_start, 0);
                assert_eq!(layer_end, 32);
            }
            _ => panic!("Expected StreamOpen variant"),
        }
    }

    #[test]
    fn test_hyphae_message_stream_data() {
        let latent = LatentVector::from_vec(vec![1.0, 2.0, 3.0], 10, Uuid::new_v4());
        let msg = HyphaeMessage::StreamData {
            stream_id: Uuid::new_v4(),
            sequence: 100,
            latent,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: HyphaeMessage = serde_json::from_str(&json).unwrap();

        match decoded {
            HyphaeMessage::StreamData {
                sequence,
                latent: l,
                ..
            } => {
                assert_eq!(sequence, 100);
                assert_eq!(l.layer_idx, 10);
                assert_eq!(l.data, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected StreamData variant"),
        }
    }

    #[test]
    fn test_hyphae_message_stream_ack() {
        let msg = HyphaeMessage::StreamAck {
            stream_id: Uuid::new_v4(),
            sequence: 50,
            received_count: 48,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: HyphaeMessage = serde_json::from_str(&json).unwrap();

        match decoded {
            HyphaeMessage::StreamAck {
                sequence,
                received_count,
                ..
            } => {
                assert_eq!(sequence, 50);
                assert_eq!(received_count, 48);
            }
            _ => panic!("Expected StreamAck variant"),
        }
    }

    #[test]
    fn test_hyphae_message_stream_close() {
        let msg = HyphaeMessage::StreamClose {
            stream_id: Uuid::new_v4(),
            reason: "completed".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let decoded: HyphaeMessage = serde_json::from_str(&json).unwrap();

        match decoded {
            HyphaeMessage::StreamClose { reason, .. } => {
                assert_eq!(reason, "completed");
            }
            _ => panic!("Expected StreamClose variant"),
        }
    }
}
