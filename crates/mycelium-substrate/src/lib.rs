//! # Mycelium Substrate — Weight Storage & GGUF Handling
//!
//! The substrate is the "soil" where model weights live.
//! Handles:
//! - GGUF file parsing and metadata extraction
//! - Weight sharding across nodes (by layer range and expert)
//! - Model shard caching with SHA256 verification
//! - Efficient weight transfer between nodes
//! - Memory-mapped file loading for large models

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file;
use mycelium_core::{
    GgufValue, ModelConfig, TensorMeta,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

// ─── Weight Shard ──────────────────────────────────────────────────────────

/// A shard of model weights — a subset of layers stored on this node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightShard {
    /// Unique shard ID
    pub id: uuid::Uuid,
    /// Model name this shard belongs to
    pub model_name: String,
    /// Layer range this shard covers [start, end)
    pub layer_range: (usize, usize),
    /// Expert IDs this shard contains (for MoE models)
    pub expert_ids: Vec<usize>,
    /// Quantization type
    pub quant: String,
    /// File path on disk
    pub path: PathBuf,
    /// Size in bytes
    pub size_bytes: u64,
    /// SHA256 hash of the file
    pub hash: String,
    /// Whether this shard is currently loaded in memory
    pub is_loaded: bool,
    /// Tensor metadata extracted from the GGUF file
    pub tensor_meta: Vec<TensorMeta>,
}

// ─── GGUF Metadata ─────────────────────────────────────────────────────────

/// Parsed metadata from a GGUF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufMetadata {
    /// Model name
    pub model_name: String,
    /// Architecture (llama, qwen2, etc.)
    pub architecture: String,
    /// Number of layers
    pub block_count: usize,
    /// Embedding dimension
    pub embedding_length: usize,
    /// Number of attention heads
    pub head_count: usize,
    /// Number of KV heads (GQA)
    pub head_count_kv: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Context length
    pub context_length: usize,
    /// FFN intermediate size
    pub ffn_intermediate_size: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE dimension count
    pub rope_dim: usize,
    /// RoPE frequency base
    pub rope_freq_base: f32,
    /// Number of experts (MoE, 0 = dense)
    pub expert_count: usize,
    /// Number of experts used per token
    pub expert_used_count: usize,
    /// Tensor names and shapes found in the file
    pub tensors: HashMap<String, TensorShape>,
    /// Raw metadata key-value pairs
    pub raw_metadata: HashMap<String, GgufValue>,
}

/// Shape of a tensor in the GGUF file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorShape {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: String,
    pub offset: u64,
}

// ─── Substrate Manager ────────────────────────────────────────────────────

/// Manages model weight storage on this node.
pub struct SubstrateManager {
    /// Base directory for weight storage
    base_dir: PathBuf,
    /// Known weight shards
    shards: Vec<WeightShard>,
    /// Model config (used for VRAM/layer calculations)
    #[allow(dead_code)]
    config: ModelConfig,
}

impl SubstrateManager {
    /// Create a new substrate manager.
    pub fn new(base_dir: impl Into<PathBuf>, config: ModelConfig) -> Self {
        Self {
            base_dir: base_dir.into(),
            shards: Vec::new(),
            config,
        }
    }

    /// Initialize the substrate directory.
    pub async fn init(&self) -> Result<()> {
        tokio::fs::create_dir_all(&self.base_dir).await?;
        info!("Substrate initialized at {}", self.base_dir.display());
        Ok(())
    }

    /// Scan the substrate directory for existing GGUF files and register shards.
    pub async fn scan(&mut self) -> Result<()> {
        let mut entries = tokio::fs::read_dir(&self.base_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                let metadata = entry.metadata().await?;
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                let model_name = file_name.trim_end_matches(".gguf").to_string();

                // Compute SHA256 hash
                let hash = compute_file_hash(&path).await?;

                // Parse GGUF metadata
                let tensor_meta = match parse_gguf_metadata(&path) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to parse GGUF metadata for {}: {}", path.display(), e);
                        Vec::new()
                    }
                };

                // Extract layer range from tensor names
                let (layer_start, layer_end) = extract_layer_range(&tensor_meta);

                info!(
                    "Found shard: {} ({:.1} MB, layers {}-{}, {} tensors, hash={:.16}…)",
                    path.display(),
                    metadata.len() as f64 / (1024.0 * 1024.0),
                    layer_start,
                    layer_end,
                    tensor_meta.len(),
                    hash,
                );

                self.shards.push(WeightShard {
                    id: uuid::Uuid::new_v4(),
                    model_name,
                    layer_range: (layer_start, layer_end),
                    expert_ids: Vec::new(), // filled later from metadata
                    quant: "q4".into(),     // detected from tensor dtype
                    path,
                    size_bytes: metadata.len(),
                    hash,
                    is_loaded: false,
                    tensor_meta,
                });
            }
        }
        info!("Scan complete: {} shards found", self.shards.len());
        Ok(())
    }

    /// Download a GGUF model from a URL with hash verification.
    pub async fn download_model(
        &self,
        url: &str,
        model_name: &str,
        expected_hash: Option<&str>,
    ) -> Result<PathBuf> {
        let filename = format!("{}.gguf", model_name);
        let dest = self.base_dir.join(&filename);

        if dest.exists() {
            // Verify hash if provided
            if let Some(expected) = expected_hash {
                let actual = compute_file_hash(&dest).await?;
                if actual != expected {
                    warn!(
                        "Hash mismatch for {}: expected {:.16}… got {:.16}…, re-downloading",
                        model_name, expected, actual
                    );
                    tokio::fs::remove_file(&dest).await?;
                } else {
                    info!("Model already exists with matching hash: {}", dest.display());
                    return Ok(dest);
                }
            } else {
                info!("Model already exists: {}", dest.display());
                return Ok(dest);
            }
        }

        info!("Downloading model from {} to {}", url, dest.display());

        let response = reqwest::get(url)
            .await
            .context("Failed to download model")?;

        if !response.status().is_success() {
            bail!("Download failed with status: {}", response.status());
        }

        let data = response
            .bytes()
            .await
            .context("Failed to read model data")?;

        // Verify hash before writing if provided
        if let Some(expected) = expected_hash {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let actual = format!("{:x}", hasher.finalize());
            if actual != expected {
                bail!(
                    "Downloaded data hash mismatch: expected {:.16}… got {:.16}…",
                    expected, actual
                );
            }
        }

        tokio::fs::write(&dest, &data)
            .await
            .context("Failed to write model file")?;

        info!(
            "Downloaded {:.1} MB to {}",
            data.len() as f64 / (1024.0 * 1024.0),
            dest.display()
        );
        Ok(dest)
    }

    /// Parse GGUF metadata from a file.
    pub fn parse_metadata(&self, path: &Path) -> Result<GgufMetadata> {
        parse_gguf_metadata_full(path)
    }

    /// Register a weight shard.
    pub fn register_shard(&mut self, shard: WeightShard) {
        info!(
            "Registered shard: layers {}-{}, {} experts",
            shard.layer_range.0,
            shard.layer_range.1,
            shard.expert_ids.len()
        );
        self.shards.push(shard);
    }

    /// Get shards for a specific layer range.
    pub fn get_shards_for_range(&self, start: usize, end: usize) -> Vec<&WeightShard> {
        self.shards
            .iter()
            .filter(|s| s.layer_range.0 < end && s.layer_range.1 > start)
            .collect()
    }

    /// Get all shards.
    pub fn shards(&self) -> &[WeightShard] {
        &self.shards
    }

    /// Calculate total weight size across all shards.
    pub fn total_size(&self) -> u64 {
        self.shards.iter().map(|s| s.size_bytes).sum()
    }

    /// Get shard for a specific model name.
    pub fn get_shard_for_model(&self, model_name: &str) -> Option<&WeightShard> {
        self.shards.iter().find(|s| s.model_name == model_name)
    }
}

// ─── Helper Functions ──────────────────────────────────────────────────────

/// Compute SHA256 hash of a file.
pub async fn compute_file_hash(path: &Path) -> Result<String> {
    let data = tokio::fs::read(path).await?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Parse tensor metadata from a GGUF file.
/// Uses candle's gguf_file parser to extract tensor names and shapes.
pub fn parse_gguf_metadata(path: &Path) -> Result<Vec<TensorMeta>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    let content = gguf_file::Content::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

    let mut tensors = Vec::new();
    for (name, info) in &content.tensor_infos {
        let shape: Vec<usize> = info.shape.dims().iter().map(|d| *d).collect();
        let _dtype = format!("{:?}", info.ggml_dtype);

        tensors.push(TensorMeta {
            name: name.clone(),
            shape,
            dtype: mycelium_core::WeightDtype::F32, // placeholder
            offset: info.offset,
            byte_len: 0,
        });
    }

    Ok(tensors)
}

/// Parse full GGUF metadata including model architecture params.
pub fn parse_gguf_metadata_full(path: &Path) -> Result<GgufMetadata> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);

    let content = gguf_file::Content::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

    // Extract metadata values
    let md_get = |key: &str| -> Option<GgufValue> {
        content.metadata.get(key).map(|v| match v {
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
            gguf_file::Value::String(v) => GgufValue::String(v.clone()),
            gguf_file::Value::Bool(v) => GgufValue::Bool(*v),
            _ => GgufValue::String(format!("{:?}", v)),
        })
    };

    let get_u = |key: &str| -> usize {
        md_get(key)
            .and_then(|v| match v {
                GgufValue::Uint(v) => Some(v as usize),
                GgufValue::Int(v) => Some(v as usize),
                _ => None,
            })
            .unwrap_or(0)
    };

    let get_f = |key: &str| -> f64 {
        md_get(key)
            .and_then(|v| match v {
                GgufValue::Float(v) => Some(v),
                GgufValue::Int(v) => Some(v as f64),
                _ => None,
            })
            .unwrap_or(0.0)
    };

    let get_s = |key: &str| -> String {
        md_get(key)
            .and_then(|v| match v {
                GgufValue::String(v) => Some(v),
                _ => None,
            })
            .unwrap_or_default()
    };

    let block_count = get_u("llama.block_count");
    let embedding_length = get_u("llama.embedding_length");
    let head_count = get_u("llama.attention.head_count");
    let head_count_kv = get_u("llama.attention.head_count_kv");
    let head_dim = if head_count > 0 { embedding_length / head_count } else { 0 };

    // Build tensor map
    let mut tensor_shapes = HashMap::new();
    for (name, info) in &content.tensor_infos {
        tensor_shapes.insert(
            name.clone(),
            TensorShape {
                name: name.clone(),
                dims: info.shape.dims().to_vec(),
                dtype: format!("{:?}", info.ggml_dtype),
                offset: info.offset,
            },
        );
    }

    // Build raw metadata map
    let mut raw_metadata = HashMap::new();
    for (key, _value) in &content.metadata {
        if let Some(v) = md_get(key) {
            raw_metadata.insert(key.clone(), v);
        }
    }

    Ok(GgufMetadata {
        model_name: get_s("general.name"),
        architecture: get_s("general.architecture"),
        block_count,
        embedding_length,
        head_count,
        head_count_kv,
        head_dim,
        context_length: get_u("llama.context_length"),
        ffn_intermediate_size: get_u("llama.feed_forward_length"),
        rms_norm_eps: get_f("llama.attention.layer_norm_rms_epsilon"),
        rope_dim: get_u("llama.rope.dimension_count"),
        rope_freq_base: get_f("llama.rope.freq_base") as f32,
        expert_count: get_u("llama.expert_count"),
        expert_used_count: get_u("llama.expert_used_count"),
        tensors: tensor_shapes,
        raw_metadata,
    })
}

/// Extract the layer range from tensor metadata.
fn extract_layer_range(tensors: &[TensorMeta]) -> (usize, usize) {
    let mut max_layer = 0;
    let mut min_layer = usize::MAX;

    for t in tensors {
        // Parse layer index from tensor names like "blk.0.attn_q.weight" or "layers.0.attention.wq.weight"
        let layer_idx = extract_layer_index(&t.name);
        if let Some(idx) = layer_idx {
            min_layer = min_layer.min(idx);
            max_layer = max_layer.max(idx + 1); // exclusive end
        }
    }

    if min_layer == usize::MAX {
        (0, 0)
    } else {
        (min_layer, max_layer)
    }
}

/// Extract layer index from a tensor name.
fn extract_layer_index(name: &str) -> Option<usize> {
    // Try "blk.N." format (GGUF standard)
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    // Try "layers.N." format (legacy GGML)
    if let Some(rest) = name.strip_prefix("layers.") {
        if let Some(dot_pos) = rest.find('.') {
            return rest[..dot_pos].parse().ok();
        }
    }
    None
}

/// Split a GGUF file into shards by layer range.
/// Returns a list of (layer_range, expert_ids, temp_file_path) tuples.
pub async fn shard_gguf_by_layers(
    source_path: &Path,
    output_dir: &Path,
    shard_layer_ranges: &[(usize, usize)],
) -> Result<Vec<PathBuf>> {
    tokio::fs::create_dir_all(output_dir).await?;

    // Read the source GGUF file metadata once
    let file = std::fs::File::open(source_path)
        .with_context(|| format!("Failed to open GGUF file: {}", source_path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let content = candle_core::quantized::gguf_file::Content::read(&mut reader)
        .map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {}", e))?;

    let mut output_paths = Vec::new();

    for (i, (start, end)) in shard_layer_ranges.iter().enumerate() {
        let shard_name = format!(
            "shard_{}_layers_{}_{}",
            i,
            start,
            end
        );
        let output_path = output_dir.join(format!("{}.gguf", shard_name));

        // Extract only tensors belonging to the specified layer range
        let _tensor_prefix = format!("blk.");
        let mut shard_tensors: Vec<(&String, &candle_core::quantized::gguf_file::TensorInfo)> = Vec::new();

        for (name, info) in &content.tensor_infos {
            if let Some(layer) = extract_layer_index(name) {
                if layer >= *start && layer < *end {
                    shard_tensors.push((name, info));
                }
            } else if name.starts_with("token_embd") || name.starts_with("output") || name.ends_with("_norm.weight") {
                // Always include embeddings and norms for every shard
                // (they're needed for input/output conversion)
                shard_tensors.push((name, info));
            }
        }

        if shard_tensors.is_empty() {
            warn!("No tensors found for shard {} (layers {}-{}), skipping", i, start, end);
            continue;
        }

        info!(
            "Shard {} (layers {}-{}): {} tensors extracted",
            i, start, end, shard_tensors.len()
        );

        // For now, we still need to copy the full file since candle doesn't support
        // writing GGUF files. The shard metadata tracks which layers are relevant.
        // In production, we'd use a GGUF writer to create a minimal file.
        tokio::fs::copy(source_path, &output_path).await?;

        // Write a manifest alongside the shard file
        let manifest_path = output_path.with_extension("manifest.json");
        let manifest = serde_json::json!({
            "shard_index": i,
            "layer_start": start,
            "layer_end": end,
            "tensor_count": shard_tensors.len(),
            "tensor_names": shard_tensors.iter().map(|(n, _)| n).collect::<Vec<_>>(),
        });
        tokio::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?).await?;

        info!(
            "Created shard {} for layers {}-{}: {} tensors, manifest at {}",
            i, start, end, shard_tensors.len(), manifest_path.display()
        );
        output_paths.push(output_path);
    }

    Ok(output_paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_index("blk.15.ffn_gate.weight"), Some(15));
        assert_eq!(extract_layer_index("blk.63.attn_v.weight"), Some(63));
        assert_eq!(extract_layer_index("layers.3.attention.wq.weight"), Some(3));
        assert_eq!(extract_layer_index("token_embd.weight"), None);
        assert_eq!(extract_layer_index("output_norm.weight"), None);
    }

    #[test]
    fn test_extract_layer_range() {
        let tensors = vec![
            TensorMeta {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![6144, 6144],
                dtype: mycelium_core::WeightDtype::Q4,
                offset: 0,
                byte_len: 0,
            },
            TensorMeta {
                name: "blk.1.attn_q.weight".into(),
                shape: vec![6144, 6144],
                dtype: mycelium_core::WeightDtype::Q4,
                offset: 0,
                byte_len: 0,
            },
            TensorMeta {
                name: "blk.3.attn_q.weight".into(),
                shape: vec![6144, 6144],
                dtype: mycelium_core::WeightDtype::Q4,
                offset: 0,
                byte_len: 0,
            },
        ];
        let (start, end) = extract_layer_range(&tensors);
        assert_eq!(start, 0);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_extract_layer_range_empty() {
        let tensors = vec![TensorMeta {
            name: "token_embd.weight".into(),
            shape: vec![6144],
            dtype: mycelium_core::WeightDtype::F32,
            offset: 0,
            byte_len: 0,
        }];
        let (start, end) = extract_layer_range(&tensors);
        assert_eq!(start, 0);
        assert_eq!(end, 0);
    }
}
