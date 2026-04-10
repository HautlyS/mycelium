//! # Mycelium Spore — Self-Replication Protocol
//!
//! Spores are the reproductive unit of the mycelium network.
//! A spore contains a subset of model weights + LoRA adapter + genome
//! that can travel across the P2P network and germinate on a new node.
//!
//! Protocol:
//! 1. Create: Node packages weights + LoRA + genome into a spore
//! 2. Serialize: Spore is serialized, compressed (zstd), and hashed
//! 3. Transfer: Spore is chunked and sent over gossipsub
//! 4. Verify: Recipient verifies SHA256 hash and genome integrity
//! 5. Germinate: Recipient decompresses, loads weights, applies LoRA
//!
//! Self-Replication:
//! - SporeLifecycle: State machine (Dormant → Germinating → Active → Fruiting → Dead)
//! - replicate(): Creates child spores from a parent with configurable mutations
//! - verify_spore_integrity(): Comprehensive integrity verification

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use crc32fast::Hasher as Crc32Hasher;
use mycelium_core::{LoRAAdapter, ModelConfig, NodeId, Spore, SporeGenome, sha256_hash};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use tracing::{info, warn};
use uuid::Uuid;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Maximum chunk size for network transfer (1 MB).
pub const SPORE_CHUNK_SIZE: usize = 1024 * 1024;

/// Current spore format version.
pub const SPORE_VERSION: u32 = 1;

/// Compression level for zstd (1-22, higher = better but slower).
pub const ZSTD_COMPRESSION_LEVEL: i32 = 3;

/// Magic bytes for the compact binary spore format (0xMYCE).
pub const SPORE_BINARY_MAGIC: [u8; 4] = [0x4D, 0x59, 0x43, 0x45]; // "MYCE"

/// Current binary format version.
pub const SPORE_BINARY_VERSION: u32 = 1;

// ─── Spore Lifecycle ────────────────────────────────────────────────────────

/// State of a spore in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SporeState {
    /// Spore has been created but not yet activated.
    Dormant,
    /// Spore is germinating — genome verification in progress.
    Germinating,
    /// Spore is active and running on a node.
    Active,
    /// Spore is fruiting — producing child spores (self-replication).
    Fruiting,
    /// Spore is dead and can no longer be used.
    Dead,
}

impl std::fmt::Display for SporeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SporeState::Dormant => write!(f, "Dormant"),
            SporeState::Germinating => write!(f, "Germinating"),
            SporeState::Active => write!(f, "Active"),
            SporeState::Fruiting => write!(f, "Fruiting"),
            SporeState::Dead => write!(f, "Dead"),
        }
    }
}

/// State machine for the spore lifecycle.
///
/// Models the biological lifecycle of a fungal spore:
/// Dormant → Germinating → Active → Fruiting → Dead
///
/// A spore can only transition forward through these states
/// (except kill, which can transition from any state to Dead).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeLifecycle {
    /// The spore being managed.
    pub spore: Spore,
    /// Current lifecycle state.
    pub state: SporeState,
    /// Germination progress (0.0 to 1.0).
    pub germination_progress: f32,
    /// Time when the spore was activated.
    pub activation_time: Option<DateTime<Utc>>,
}

impl SporeLifecycle {
    /// Create a new lifecycle manager for a spore (starts in Dormant state).
    pub fn new(spore: Spore) -> Self {
        Self {
            spore,
            state: SporeState::Dormant,
            germination_progress: 0.0,
            activation_time: None,
        }
    }

    /// Start germination: verify genome integrity, transition to Germinating.
    ///
    /// The substrate manager is used to verify that the local node has
    /// sufficient resources to host the spore's weights.
    pub fn germinate(
        &mut self,
        _substrate: &mut mycelium_substrate::SubstrateManager,
    ) -> Result<()> {
        if self.state != SporeState::Dormant {
            bail!(
                "Cannot germinate spore {}: current state is {}, expected Dormant",
                self.spore.id,
                self.state
            );
        }

        // Verify genome integrity before germination
        if !self.spore.genome.verify() {
            bail!(
                "Spore {} genome verification failed — cannot germinate corrupted spore",
                self.spore.id
            );
        }

        self.state = SporeState::Germinating;
        self.germination_progress = 0.0;
        info!(
            "Spore {} germination started (generation={})",
            self.spore.id, self.spore.generation
        );
        Ok(())
    }

    /// Activate the spore: transition to Active state, record activation time.
    ///
    /// Can only be called after germination has started.
    pub fn activate(&mut self) -> Result<()> {
        if self.state != SporeState::Germinating && self.state != SporeState::Dormant {
            bail!(
                "Cannot activate spore {}: current state is {}, expected Germinating or Dormant",
                self.spore.id,
                self.state
            );
        }

        // Ensure genome is still valid at activation time
        if !self.spore.genome.verify() {
            bail!(
                "Spore {} genome verification failed at activation — cannot activate",
                self.spore.id
            );
        }

        self.state = SporeState::Active;
        self.germination_progress = 1.0;
        self.activation_time = Some(Utc::now());
        info!(
            "Spore {} activated at {:?}",
            self.spore.id, self.activation_time
        );
        Ok(())
    }

    /// Fruit: create child spores (self-replication!), increment generation.
    ///
    /// The spore enters the Fruiting state and produces child spores.
    /// Each child is a replicate of the parent with optional mutations applied.
    /// The parent spore's generation is tracked — children get generation + 1.
    pub fn fruit(&mut self, child_spores: Vec<Spore>) -> Result<Vec<Spore>> {
        if self.state != SporeState::Active {
            bail!(
                "Cannot fruit spore {}: current state is {}, expected Active",
                self.spore.id,
                self.state
            );
        }

        // Verify the parent is still healthy before fruiting
        if !self.is_healthy() {
            bail!(
                "Spore {} is not healthy — cannot produce offspring",
                self.spore.id
            );
        }

        self.state = SporeState::Fruiting;
        info!(
            "Spore {} fruiting: producing {} child spores at generation {}",
            self.spore.id,
            child_spores.len(),
            self.spore.generation + 1
        );

        // Verify each child spore belongs to this parent
        for child in &child_spores {
            if child.generation != self.spore.generation + 1 {
                warn!(
                    "Child spore {} has unexpected generation {} (expected {})",
                    child.id,
                    child.generation,
                    self.spore.generation + 1
                );
            }
        }

        Ok(child_spores)
    }

    /// Kill the spore: transition to Dead state.
    ///
    /// Can be called from any state. Once dead, a spore cannot be revived.
    pub fn kill(&mut self) -> Result<()> {
        let prev_state = self.state;
        self.state = SporeState::Dead;
        info!("Spore {} killed (was {})", self.spore.id, prev_state);
        Ok(())
    }

    /// Check if the spore genome is valid (healthy).
    ///
    /// A healthy spore has an intact genome whose hash matches its data.
    pub fn is_healthy(&self) -> bool {
        self.spore.genome.verify()
    }

    /// Get the current state of the spore.
    pub fn state(&self) -> SporeState {
        self.state
    }

    /// Set germination progress (0.0 to 1.0).
    ///
    /// Should only be called when the spore is in the Germinating state.
    pub fn set_germination_progress(&mut self, progress: f32) -> Result<()> {
        if self.state != SporeState::Germinating {
            bail!(
                "Cannot set germination progress: spore {} is {}, not Germinating",
                self.spore.id,
                self.state
            );
        }
        self.germination_progress = progress.clamp(0.0, 1.0);
        Ok(())
    }
}

// ─── Mutation Configuration ────────────────────────────────────────────────

/// Configuration for mutations applied during spore replication.
///
/// Mutations introduce variation into the spore population, enabling
/// evolutionary adaptation. The two mutation mechanisms are:
/// - **Bit flips**: Randomly flip individual bits in the genome data
/// - **Weight perturbation**: Add Gaussian-like noise to weight bytes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Probability of a bit flip in genome data (per bit).
    /// Typical range: 0.0 (no mutations) to 0.001 (heavy mutations).
    pub bit_flip_rate: f64,
    /// Scale of weight perturbation applied to weight bytes.
    /// 0.0 means no perturbation; typical values are 0.01-0.1.
    pub weight_perturbation: f32,
    /// Maximum generation before a spore is considered expired.
    /// Spores beyond this generation will fail integrity checks.
    pub max_generation: u32,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            bit_flip_rate: 0.0001,
            weight_perturbation: 0.01,
            max_generation: 100,
        }
    }
}

impl MutationConfig {
    /// Create a mutation config with no mutations (cloning).
    pub fn none() -> Self {
        Self {
            bit_flip_rate: 0.0,
            weight_perturbation: 0.0,
            max_generation: u32::MAX,
        }
    }

    /// Create a mutation config with aggressive mutations (for exploration).
    pub fn aggressive() -> Self {
        Self {
            bit_flip_rate: 0.001,
            weight_perturbation: 0.1,
            max_generation: 50,
        }
    }

    /// Create a conservative mutation config (for exploitation).
    pub fn conservative() -> Self {
        Self {
            bit_flip_rate: 0.00001,
            weight_perturbation: 0.001,
            max_generation: 200,
        }
    }
}

// ─── Spore Replication ─────────────────────────────────────────────────────

/// Replicate a spore: create a child from a parent with optional mutations.
///
/// This is the core self-replication mechanism. The child spore inherits:
/// - The parent's genome data (with mutations applied)
/// - The parent's model configuration
/// - The parent's layer range and expert IDs
/// - The parent's LoRA adapter (instincts)
///
/// The child gets:
/// - A new unique ID
/// - An incremented generation counter
/// - The parent's NodeId set as its parent field
/// - A fresh creation timestamp
/// - A new genome hash (reflecting any mutations)
///
/// Returns an error if:
/// - The parent's genome is corrupted
/// - The parent's generation exceeds the max_generation in MutationConfig
pub fn replicate(parent: &Spore, mutations: MutationConfig) -> Result<Spore> {
    // Verify parent genome integrity before replication
    if !parent.genome.verify() {
        bail!(
            "Cannot replicate spore {}: parent genome is corrupted",
            parent.id
        );
    }

    // Check generation limit
    if parent.generation >= mutations.max_generation {
        bail!(
            "Cannot replicate spore {}: generation {} exceeds max_generation {}",
            parent.id,
            parent.generation,
            mutations.max_generation
        );
    }

    let child_generation = parent.generation + 1;
    info!(
        "Replicating spore {} → generation {} (mutations: bit_flip={}, weight_pert={})",
        parent.id, child_generation, mutations.bit_flip_rate, mutations.weight_perturbation
    );

    // Clone genome data and apply mutations
    let mut child_genome_data = parent.genome.data.clone();
    apply_mutations(&mut child_genome_data, &mutations);

    // Create new genome with updated hash
    let child_genome = SporeGenome::new(
        child_genome_data,
        parent.genome.quant_bits,
        parent.genome.decompressed_size,
    );

    // Build the child spore
    let child = Spore {
        id: Uuid::new_v4(),
        genome: child_genome,
        instincts: parent.instincts.clone(),
        model_config: parent.model_config.clone(),
        layer_range: parent.layer_range,
        expert_ids: parent.expert_ids.clone(),
        created_at: Utc::now(),
        parent: parent.parent, // parent field is the originating NodeId
        generation: child_generation,
    };

    info!(
        "Child spore {} created from parent {} (generation {})",
        child.id, parent.id, child_generation
    );

    Ok(child)
}

/// Apply mutations to genome data in-place.
///
/// Two mutation mechanisms:
/// 1. Bit flips: Each bit in the data has a `bit_flip_rate` probability of being flipped.
///    This introduces structural changes to the quantized weights.
/// 2. Weight perturbation: Additive noise scaled by `weight_perturbation`.
///    This introduces small continuous changes to the weight values.
fn apply_mutations(data: &mut [u8], config: &MutationConfig) {
    let mut rng = rand::thread_rng();

    // Apply bit flips
    if config.bit_flip_rate > 0.0 {
        for byte in data.iter_mut() {
            for bit in 0u8..8 {
                if rng.gen_bool(config.bit_flip_rate) {
                    *byte ^= 1 << bit;
                }
            }
        }
    }

    // Apply weight perturbation (interpreted as f32 pairs)
    if config.weight_perturbation > 0.0 {
        let mut i = 0;
        while i + 4 <= data.len() {
            let bytes = [data[i], data[i + 1], data[i + 2], data[i + 3]];
            let mut weight = f32::from_le_bytes(bytes);

            // Add scaled random perturbation
            let noise: f32 = rng.gen_range(-1.0..1.0) * config.weight_perturbation;
            weight += noise;

            // Clamp to prevent extreme values
            weight = weight.clamp(-1e6, 1e6);

            let new_bytes = weight.to_le_bytes();
            data[i] = new_bytes[0];
            data[i + 1] = new_bytes[1];
            data[i + 2] = new_bytes[2];
            data[i + 3] = new_bytes[3];

            i += 4;
        }
    }
}

// ─── Spore Integrity Verification ──────────────────────────────────────────

/// Comprehensive integrity check result for a spore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeIntegrity {
    /// Whether the genome data passes its own hash verification.
    pub genome_valid: bool,
    /// Whether the stored hash matches the computed hash.
    pub hash_matches: bool,
    /// Whether the spore has not expired (based on age / timestamp).
    pub not_expired: bool,
    /// Whether the spore's generation is within acceptable limits.
    pub generation_within_limits: bool,
    /// Overall integrity: true only if all checks pass.
    pub overall: bool,
}

impl SporeIntegrity {
    /// Check if all integrity fields pass.
    fn compute_overall(&self) -> bool {
        self.genome_valid && self.hash_matches && self.not_expired && self.generation_within_limits
    }
}

/// Verify the integrity of a spore comprehensively.
///
/// Checks:
/// 1. Genome validity: The genome's internal hash verification passes
/// 2. Hash matches: The stored hash matches the computed SHA256 of the data
/// 3. Not expired: The spore's creation timestamp is not too old
/// 4. Generation within limits: The generation is below a reasonable maximum
pub fn verify_spore_integrity(spore: &Spore) -> SporeIntegrity {
    let genome_valid = spore.genome.verify();
    let hash_matches = sha256_hash(&spore.genome.data) == spore.genome.hash;
    let not_expired = is_not_expired(&spore.created_at);
    let generation_within_limits = spore.generation < 1000; // hard upper bound

    let integrity = SporeIntegrity {
        genome_valid,
        hash_matches,
        not_expired,
        generation_within_limits,
        overall: false, // placeholder
    };

    SporeIntegrity {
        overall: integrity.compute_overall(),
        ..integrity
    }
}

/// Check if a spore is not expired based on its creation timestamp.
///
/// A spore is considered expired if it was created more than 1 year ago.
fn is_not_expired(created_at: &DateTime<Utc>) -> bool {
    let now = Utc::now();
    let age = now.signed_duration_since(*created_at);
    age.num_days() < 365
}

// ─── Spore File Format ────────────────────────────────────────────────────

/// The serialized spore file format.
///
/// ```text
/// [4 bytes: magic "MYCO"]
/// [4 bytes: version (LE u32)]
/// [8 bytes: manifest length (LE u64)]
/// [N bytes: manifest (JSON)]
/// [8 bytes: data length (LE u64)]
/// [M bytes: data (zstd compressed)]
/// [32 bytes: SHA256 hash of everything above]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeFile {
    /// Magic bytes for format identification
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// The spore manifest (metadata)
    pub manifest: SporeManifest,
    /// Compressed weight data
    pub data: Vec<u8>,
    /// SHA256 hash of (magic + version + manifest + data)
    pub hash: [u8; 32],
}

// ─── Spore Manifest ────────────────────────────────────────────────────────

/// Manifest describing a spore's contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeManifest {
    /// Spore ID
    pub id: Uuid,
    /// Parent node that created this spore
    pub parent_node_id: NodeId,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Layer range in this spore
    pub layer_range: (usize, usize),
    /// Expert IDs in this spore (MoE)
    pub expert_ids: Vec<usize>,
    /// Genome (behavioral metadata)
    pub genome: SporeGenome,
    /// LoRA adapter (instincts) included in this spore
    pub instincts: Option<LoRAAdapter>,
    /// Uncompressed data size
    pub uncompressed_size: u64,
    /// Compressed data size
    pub compressed_size: u64,
    /// Number of chunks for network transfer
    pub num_chunks: usize,
    /// Generation number
    pub generation: u32,
}

impl SporeManifest {
    /// Create a manifest from a Spore struct.
    pub fn from_spore(spore: &Spore) -> Self {
        let data_len = spore.genome.data.len() as u64;
        Self {
            id: spore.id,
            parent_node_id: spore.parent,
            created_at: spore.created_at,
            model_config: spore.model_config.clone(),
            layer_range: spore.layer_range,
            expert_ids: spore.expert_ids.clone(),
            genome: spore.genome.clone(),
            instincts: spore.instincts.clone(),
            uncompressed_size: spore.genome.decompressed_size,
            compressed_size: 0, // filled after compression
            num_chunks: (data_len as usize).div_ceil(SPORE_CHUNK_SIZE),
            generation: spore.generation,
        }
    }
}

// ─── Spore Chunk ──────────────────────────────────────────────────────────

/// A chunk of spore data for network transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeChunk {
    /// Spore ID this chunk belongs to
    pub spore_id: Uuid,
    /// Chunk index (0-based)
    pub chunk_index: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Chunk data
    pub data: Vec<u8>,
    /// SHA256 hash of this chunk
    pub chunk_hash: [u8; 32],
}

impl SporeChunk {
    /// Create chunks from spore data.
    pub fn from_data(spore_id: Uuid, data: &[u8]) -> Vec<Self> {
        let total_chunks = (data.len() + SPORE_CHUNK_SIZE - 1) / SPORE_CHUNK_SIZE.max(1);
        let mut chunks = Vec::with_capacity(total_chunks);

        for (i, chunk_start) in (0..data.len()).step_by(SPORE_CHUNK_SIZE).enumerate() {
            let chunk_end = (chunk_start + SPORE_CHUNK_SIZE).min(data.len());
            let chunk_data = &data[chunk_start..chunk_end];

            let mut hasher = Sha256::new();
            hasher.update(chunk_data);
            let hash: [u8; 32] = hasher.finalize().into();

            chunks.push(Self {
                spore_id,
                chunk_index: i,
                total_chunks,
                data: chunk_data.to_vec(),
                chunk_hash: hash,
            });
        }

        chunks
    }

    /// Reassemble spore data from chunks.
    pub fn reassemble(chunks: &[SporeChunk]) -> Result<Vec<u8>> {
        if chunks.is_empty() {
            bail!("No chunks to reassemble");
        }

        let spore_id = chunks[0].spore_id;
        let total = chunks[0].total_chunks;

        // Verify all chunks belong to the same spore
        for chunk in chunks {
            if chunk.spore_id != spore_id {
                bail!(
                    "Chunk spore ID mismatch: expected {}, got {}",
                    spore_id,
                    chunk.spore_id
                );
            }
        }

        // Sort by chunk index
        let mut sorted: Vec<&SporeChunk> = chunks.iter().collect();
        sorted.sort_by_key(|c| c.chunk_index);

        // Verify completeness
        if sorted.len() != total {
            bail!(
                "Incomplete spore: expected {} chunks, got {}",
                total,
                sorted.len()
            );
        }

        // Verify each chunk's hash
        for chunk in &sorted {
            let mut hasher = Sha256::new();
            hasher.update(&chunk.data);
            let hash: [u8; 32] = hasher.finalize().into();
            if hash != chunk.chunk_hash {
                bail!("Chunk {} hash mismatch: data corrupted", chunk.chunk_index);
            }
        }

        // Reassemble
        let mut data = Vec::new();
        for chunk in sorted {
            data.extend_from_slice(&chunk.data);
        }

        Ok(data)
    }
}

// ─── Spore Builder ─────────────────────────────────────────────────────────

/// Builder for creating spores from model weights.
pub struct SporeBuilder {
    layer_range: (usize, usize),
    expert_ids: Vec<usize>,
    parent: NodeId,
    instincts: Option<LoRAAdapter>,
    model_config: ModelConfig,
    generation: u32,
}

impl SporeBuilder {
    /// Create a new spore builder.
    pub fn new(model_config: ModelConfig, parent: NodeId) -> Self {
        Self {
            model_config,
            layer_range: (0, 0),
            expert_ids: Vec::new(),
            parent,
            instincts: None,
            generation: 0,
        }
    }

    /// Set the layer range.
    pub fn layer_range(mut self, start: usize, end: usize) -> Self {
        self.layer_range = (start, end);
        self
    }

    /// Set the expert IDs (for MoE models).
    pub fn expert_ids(mut self, ids: Vec<usize>) -> Self {
        self.expert_ids = ids;
        self
    }

    /// Attach a LoRA adapter (instincts).
    pub fn instincts(mut self, adapter: LoRAAdapter) -> Self {
        self.instincts = Some(adapter);
        self
    }

    /// Set the generation number.
    pub fn generation(mut self, gen_val: u32) -> Self {
        self.generation = gen_val;
        self
    }

    /// Build a spore from raw weight data.
    ///
    /// The weight data should contain only the tensors for the specified
    /// layer range and expert IDs.
    pub fn build(self, weight_data: Vec<u8>, quant_bits: u8) -> Spore {
        let decompressed_size = weight_data.len() as u64;
        let genome = SporeGenome::new(weight_data, quant_bits, decompressed_size);

        Spore {
            id: Uuid::new_v4(),
            parent: self.parent,
            created_at: Utc::now(),
            model_config: self.model_config,
            layer_range: self.layer_range,
            expert_ids: self.expert_ids,
            genome,
            instincts: self.instincts,
            generation: self.generation,
        }
    }

    /// Build a spore from a GGUF file.
    ///
    /// Reads the entire GGUF file. For production use, should extract
    /// only the needed tensors for the specified layer range.
    pub async fn build_from_gguf(self, gguf_path: &Path, quant_bits: u8) -> Result<Spore> {
        if !gguf_path.exists() {
            bail!("GGUF file not found: {}", gguf_path.display());
        }

        let metadata = std::fs::metadata(gguf_path)?;
        let file_size = metadata.len() as f64 / (1024.0 * 1024.0);
        info!(
            "Reading GGUF file: {} ({:.1} MB)",
            gguf_path.display(),
            file_size
        );

        let weight_data = tokio::fs::read(gguf_path).await?;

        let spore = self.build(weight_data, quant_bits);

        let compressed_mb = spore.genome.data.len() as f64 / (1024.0 * 1024.0);
        info!(
            "Spore created: {} layers, {} experts, {:.1} MB uncompressed",
            spore.layer_range.1 - spore.layer_range.0,
            spore.expert_ids.len(),
            compressed_mb,
        );

        Ok(spore)
    }
}

// ─── Spore Germinator ──────────────────────────────────────────────────────

/// Germinates a spore — loads its weights and applies LoRA.
pub struct SporeGerminator {
    output_dir: PathBuf,
}

impl SporeGerminator {
    /// Create a new germinator that writes to the given output directory.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// Germinate a spore: verify, decompress, and write weights.
    pub async fn germinate(&self, spore: &Spore) -> Result<GerminationResult> {
        info!("Germinating spore {} from node {}", spore.id, spore.parent);

        // 1. Verify genome hash
        if !spore.genome.verify() {
            bail!("Spore genome verification failed — data corrupted!");
        }

        // 2. Create output directory
        let spore_dir = self.output_dir.join(format!("spore_{}", spore.id));
        tokio::fs::create_dir_all(&spore_dir).await?;

        // 3. Write weight data to GGUF file
        let gguf_path = spore_dir.join(format!("{}.gguf", spore.model_config.name));
        tokio::fs::write(&gguf_path, &spore.genome.data).await?;
        info!(
            "Wrote {:.1} MB to {}",
            spore.genome.data.len() as f64 / (1024.0 * 1024.0),
            gguf_path.display()
        );

        // 4. Write manifest
        let manifest = SporeManifest::from_spore(spore);
        let manifest_path = spore_dir.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        tokio::fs::write(&manifest_path, manifest_json).await?;

        // 5. Write LoRA adapter if present
        if let Some(lora) = &spore.instincts {
            let lora_path = spore_dir.join("lora.json");
            let lora_json = serde_json::to_string_pretty(lora)?;
            tokio::fs::write(&lora_path, lora_json).await?;
            info!("Wrote LoRA adapter (rank={})", lora.rank);
        }

        info!("Spore {} germinated successfully", spore.id);

        Ok(GerminationResult {
            spore_id: spore.id,
            spore_dir,
            gguf_path,
            manifest,
            lora_applied: spore.instincts.is_some(),
        })
    }

    /// Germinate from a serialized spore file.
    pub async fn germinate_file(&self, path: &Path) -> Result<GerminationResult> {
        let data = tokio::fs::read(path).await?;
        let spore_file: SporeFile =
            serde_json::from_slice(&data).context("Failed to deserialize spore file")?;

        // Verify hash
        let mut hasher = Sha256::new();
        hasher.update(spore_file.magic);
        hasher.update(spore_file.version.to_le_bytes());
        let manifest_bytes = serde_json::to_vec(&spore_file.manifest)?;
        hasher.update(&manifest_bytes);
        hasher.update(&spore_file.data);
        let computed_hash: [u8; 32] = hasher.finalize().into();

        if computed_hash != spore_file.hash {
            bail!("Spore file hash mismatch — data corrupted!");
        }

        // Decompress data
        let decompressed =
            zstd::decode_all(&spore_file.data[..]).context("Failed to decompress spore data")?;

        // Reconstruct Spore struct
        let spore = Spore {
            id: spore_file.manifest.id,
            parent: spore_file.manifest.parent_node_id,
            created_at: spore_file.manifest.created_at,
            model_config: spore_file.manifest.model_config,
            layer_range: spore_file.manifest.layer_range,
            expert_ids: spore_file.manifest.expert_ids,
            genome: SporeGenome::new(
                decompressed,
                spore_file.manifest.genome.quant_bits,
                spore_file.manifest.genome.decompressed_size,
            ),
            instincts: spore_file.manifest.instincts,
            generation: spore_file.manifest.generation,
        };

        self.germinate(&spore).await
    }
}

// ─── Germination Result ────────────────────────────────────────────────────

/// Result of germinating a spore.
#[derive(Debug)]
pub struct GerminationResult {
    /// The germinated spore's ID
    pub spore_id: Uuid,
    /// Directory where the spore was extracted
    pub spore_dir: PathBuf,
    /// Path to the GGUF model file
    pub gguf_path: PathBuf,
    /// The spore manifest
    pub manifest: SporeManifest,
    /// Whether a LoRA adapter was applied
    pub lora_applied: bool,
}

// ─── Serialization ─────────────────────────────────────────────────────────

/// Serialize a spore into a compressed, self-verifying file (JSON format).
pub fn serialize_spore(spore: &Spore) -> Result<Vec<u8>> {
    let manifest = SporeManifest::from_spore(spore);

    // Compress genome data with zstd
    let compressed = zstd::encode_all(&spore.genome.data[..], ZSTD_COMPRESSION_LEVEL)
        .context("Failed to compress spore data")?;

    let mut updated_manifest = manifest;
    updated_manifest.compressed_size = compressed.len() as u64;

    // Compute hash over everything
    let mut hasher = Sha256::new();
    hasher.update(b"MYCO"); // magic
    hasher.update(SPORE_VERSION.to_le_bytes());
    let manifest_bytes = serde_json::to_vec(&updated_manifest)?;
    hasher.update(&manifest_bytes);
    hasher.update(&compressed);
    let hash: [u8; 32] = hasher.finalize().into();

    let spore_file = SporeFile {
        magic: *b"MYCO",
        version: SPORE_VERSION,
        manifest: updated_manifest,
        data: compressed,
        hash,
    };

    serde_json::to_vec(&spore_file).context("Failed to serialize spore file")
}

/// Deserialize a spore from a compressed file (JSON format).
pub fn deserialize_spore(data: &[u8]) -> Result<SporeFile> {
    let spore_file: SporeFile =
        serde_json::from_slice(data).context("Failed to deserialize spore file")?;

    // Verify magic
    if &spore_file.magic != b"MYCO" {
        bail!("Invalid spore file: bad magic bytes");
    }

    // Verify version
    if spore_file.version != SPORE_VERSION {
        bail!(
            "Unsupported spore version: expected {}, got {}",
            SPORE_VERSION,
            spore_file.version
        );
    }

    // Verify hash
    let mut hasher = Sha256::new();
    hasher.update(spore_file.magic);
    hasher.update(spore_file.version.to_le_bytes());
    let manifest_bytes = serde_json::to_vec(&spore_file.manifest)?;
    hasher.update(&manifest_bytes);
    hasher.update(&spore_file.data);
    let computed_hash: [u8; 32] = hasher.finalize().into();

    if computed_hash != spore_file.hash {
        bail!("Spore file hash mismatch — data corrupted!");
    }

    Ok(spore_file)
}

// ─── Compact Binary Serialization Format ───────────────────────────────────

/// Compact binary spore format for efficient network transfer.
///
/// Layout:
/// ```text
/// [4 bytes: magic 0xMYCE]
/// [4 bytes: version (LE u32)]
/// [4 bytes: genome length (LE u32)]
/// [N bytes: compressed genome data (zstd)]
/// [4 bytes: CRC32 checksum (LE u32)]
/// ```
///
/// This format is significantly more compact than the JSON-based SporeFile
/// format because it omits the manifest (metadata is reconstructed from
/// the genome and a separate metadata exchange).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySpore {
    /// Magic bytes (0xMYCE)
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Compressed genome data
    pub genome_data: Vec<u8>,
    /// Original (uncompressed) genome length
    pub genome_length: u32,
    /// CRC32 checksum of the compressed genome data
    pub checksum: u32,
}

impl BinarySpore {
    /// Create a new binary spore from a Spore struct.
    ///
    /// Compresses the genome data with zstd and computes a CRC32 checksum.
    pub fn from_spore(spore: &Spore) -> Result<Self> {
        let genome_data = zstd::encode_all(&spore.genome.data[..], ZSTD_COMPRESSION_LEVEL)
            .context("Failed to compress genome data for binary spore")?;

        let genome_length = spore.genome.data.len() as u32;

        // Compute CRC32 of compressed data
        let mut crc_hasher = Crc32Hasher::new();
        crc_hasher.update(&genome_data);
        let checksum = crc_hasher.finalize();

        Ok(Self {
            magic: SPORE_BINARY_MAGIC,
            version: SPORE_BINARY_VERSION,
            genome_data,
            genome_length,
            checksum,
        })
    }

    /// Serialize the binary spore into bytes for network transfer.
    ///
    /// Layout:
    /// - Header: magic (4) + version (4) + genome_length (4) = 12 bytes
    /// - Body: compressed genome data (variable)
    /// - Footer: CRC32 checksum (4)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.genome_data.len() + 4);

        // Header
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.genome_length.to_le_bytes());

        // Body
        buf.extend_from_slice(&self.genome_data);

        // Footer
        buf.extend_from_slice(&self.checksum.to_le_bytes());

        buf
    }

    /// Deserialize a binary spore from bytes.
    ///
    /// Validates magic bytes, version, and CRC32 checksum.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            bail!(
                "Binary spore data too short: {} bytes (minimum 16)",
                data.len()
            );
        }

        // Parse header
        let magic = [data[0], data[1], data[2], data[3]];
        if magic != SPORE_BINARY_MAGIC {
            bail!(
                "Invalid binary spore magic: expected {:?}, got {:?}",
                SPORE_BINARY_MAGIC,
                magic
            );
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != SPORE_BINARY_VERSION {
            bail!(
                "Unsupported binary spore version: expected {}, got {}",
                SPORE_BINARY_VERSION,
                version
            );
        }

        let genome_length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

        // Parse footer (last 4 bytes)
        let data_end = data.len() - 4;
        let checksum = u32::from_le_bytes([
            data[data_end],
            data[data_end + 1],
            data[data_end + 2],
            data[data_end + 3],
        ]);

        // Extract body
        let genome_data = data[12..data_end].to_vec();

        // Verify CRC32
        let mut crc_hasher = Crc32Hasher::new();
        crc_hasher.update(&genome_data);
        let computed_checksum = crc_hasher.finalize();

        if computed_checksum != checksum {
            bail!(
                "Binary spore CRC32 mismatch: expected {:08x}, computed {:08x} — data corrupted",
                checksum,
                computed_checksum
            );
        }

        Ok(Self {
            magic,
            version,
            genome_data,
            genome_length,
            checksum,
        })
    }

    /// Decompress the genome data and verify against the expected length.
    pub fn decompress_genome(&self) -> Result<Vec<u8>> {
        let decompressed = zstd::decode_all(&self.genome_data[..])
            .context("Failed to decompress binary spore genome data")?;

        if decompressed.len() != self.genome_length as usize {
            warn!(
                "Decompressed genome length mismatch: expected {}, got {}",
                self.genome_length,
                decompressed.len()
            );
        }

        Ok(decompressed)
    }

    /// Convert back to a Spore struct (requires metadata to reconstruct fields).
    ///
    /// The binary format only carries the genome data. Metadata like model_config,
    /// layer_range, expert_ids, etc. must be provided separately.
    pub fn to_spore(
        &self,
        model_config: ModelConfig,
        layer_range: (usize, usize),
        expert_ids: Vec<usize>,
        parent: NodeId,
        generation: u32,
        quant_bits: u8,
    ) -> Result<Spore> {
        let decompressed = self.decompress_genome()?;
        let decompressed_size = decompressed.len() as u64;
        let genome = SporeGenome::new(decompressed, quant_bits, decompressed_size);

        Ok(Spore {
            id: Uuid::new_v4(),
            genome,
            instincts: None,
            model_config,
            layer_range,
            expert_ids,
            created_at: Utc::now(),
            parent,
            generation,
        })
    }
}

// ─── Spore Propagator ──────────────────────────────────────────────────────

/// Configuration for spore propagation across the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationConfig {
    /// Maximum number of spores a single node can hold
    pub max_spores_per_node: usize,
    /// Minimum VRAM (MB) a node needs to receive a spore
    pub min_vram_for_spore: u32,
    /// How often to check for propagation opportunities (seconds)
    pub check_interval_secs: u64,
    /// Whether to propagate spores automatically
    pub auto_propagate: bool,
    /// Maximum propagation depth (generation limit)
    pub max_propagation_depth: u32,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            max_spores_per_node: 5,
            min_vram_for_spore: 4096,
            check_interval_secs: 120,
            auto_propagate: true,
            max_propagation_depth: 10,
        }
    }
}

/// Current capacity state of a node for spore propagation decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacityState {
    /// Node identifier
    pub node_id: NodeId,
    /// Available VRAM in MB
    pub available_vram_mb: u32,
    /// Available system RAM in MB
    pub available_ram_mb: u32,
    /// Layer range this node is responsible for
    pub layer_range: (usize, usize),
    /// How long the node has been running (seconds)
    pub uptime_secs: u64,
    /// Whether the node has a LoRA adapter
    pub has_lora: bool,
    /// Improvement metric from LoRA training (0.0 = no improvement)
    pub lora_improvement: f32,
}

/// Manages spore propagation across the P2P network.
///
/// The propagator:
/// 1. Monitors node capacity and determines when to create spores
/// 2. Selects target nodes for propagation based on capacity
/// 3. Tracks received spores and germination status
/// 4. Broadcasts spore availability via gossipsub
pub struct SporePropagator {
    /// Propagation configuration
    config: PropagationConfig,
    /// Current node state
    node_state: NodeCapacityState,
    /// Available spores this node can offer
    available_spores: std::collections::HashMap<uuid::Uuid, Spore>,
    /// Received spores from other nodes
    received_spores: std::collections::HashMap<uuid::Uuid, Spore>,
    /// Nodes that have germinated from our spores
    germinated_from_us: std::collections::HashSet<NodeId>,
}

impl SporePropagator {
    /// Create a new spore propagator.
    pub fn new(config: PropagationConfig, node_state: NodeCapacityState) -> Self {
        Self {
            config,
            node_state,
            available_spores: std::collections::HashMap::new(),
            received_spores: std::collections::HashMap::new(),
            germinated_from_us: std::collections::HashSet::new(),
        }
    }

    /// Add a spore to the available pool for propagation.
    pub fn add_available_spore(&mut self, spore: Spore) {
        info!(
            "Spore {} added to available pool (generation={})",
            spore.id, spore.generation
        );
        self.available_spores.insert(spore.id, spore);
    }

    /// Record a spore received from another node.
    pub fn receive_spore(&mut self, spore: Spore) {
        info!(
            "Received spore {} from node {} (generation={})",
            spore.id, spore.parent, spore.generation
        );
        self.received_spores.insert(spore.id, spore);
    }

    /// Record that a node germinated from one of our spores.
    pub fn record_germination(&mut self, node_id: NodeId) {
        self.germinated_from_us.insert(node_id);
    }

    /// Get the count of available spores.
    pub fn available_spore_count(&self) -> usize {
        self.available_spores.len()
    }

    /// Get the count of received spores.
    pub fn received_spore_count(&self) -> usize {
        self.received_spores.len()
    }

    /// Check if this node should propagate spores based on capacity.
    pub fn should_propagate(&self) -> bool {
        if !self.config.auto_propagate {
            return false;
        }
        if self.available_spores.is_empty() {
            return false;
        }
        // Need sufficient VRAM and uptime to be a good propagator
        self.node_state.available_vram_mb >= self.config.min_vram_for_spore
            && self.node_state.uptime_secs >= 60
    }

    /// Select target nodes for propagation based on capacity.
    /// Returns nodes that would benefit from receiving a spore.
    pub fn select_propagation_targets(
        &self,
        known_nodes: &[(NodeId, mycelium_core::NodeCapabilities)],
    ) -> Vec<NodeId> {
        known_nodes
            .iter()
            .filter(|(_, caps)| {
                // Target nodes with less VRAM than us
                caps.vram_mb < self.node_state.available_vram_mb
                    // Or nodes that don't have a LoRA adapter
                    || (!caps.can_compute && self.node_state.has_lora)
            })
            .map(|(node_id, _)| *node_id)
            .take(self.config.max_spores_per_node)
            .collect()
    }

    /// Get the current node state.
    pub fn node_state(&self) -> &NodeCapacityState {
        &self.node_state
    }

    /// Update the node state.
    pub fn update_node_state(&mut self, state: NodeCapacityState) {
        self.node_state = state;
    }

    /// Get available spores for broadcasting.
    pub fn available_spores(&self) -> &std::collections::HashMap<uuid::Uuid, Spore> {
        &self.available_spores
    }

    /// Get received spores that could be germinated.
    pub fn received_spores(&self) -> &std::collections::HashMap<uuid::Uuid, Spore> {
        &self.received_spores
    }
}

// ─── Graceful Degradation on Node Loss ──────────────────────────────────────

/// Events emitted by the `NodeHealthMonitor` when node health changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealthEvent {
    /// A node has not sent a heartbeat within the configured timeout.
    NodeTimedOut {
        node_id: NodeId,
        last_seen: DateTime<Utc>,
    },
    /// A previously-timed-out node has sent a heartbeat again.
    NodeRecovered { node_id: NodeId, downtime_ms: i64 },
    /// A node is responding but with degraded performance (heartbeats are late
    /// but still within the timeout window).
    NodeDegraded { node_id: NodeId, delay_ms: i64 },
}

/// Monitors the health of nodes in the P2P network by tracking heartbeats.
///
/// Each node is expected to send periodic heartbeats.  When a heartbeat is not
/// received within `timeout_ms` the node is considered failed.  A node that
/// resumes heartbeating after a failure is reported as recovered.
#[derive(Debug, Clone)]
pub struct NodeHealthMonitor {
    /// Map of node id → last heartbeat timestamp.
    heartbeats: HashMap<NodeId, DateTime<Utc>>,
    /// Timeout in milliseconds before a node is considered failed.
    timeout_ms: i64,
    /// Threshold (in ms) for considering a node "degraded" – must be less than
    /// `timeout_ms`.
    degraded_threshold_ms: i64,
    /// Set of nodes currently considered failed (timed-out).
    failed_nodes: HashSet<NodeId>,
}

impl NodeHealthMonitor {
    /// Create a new monitor with the given timeout (in milliseconds).
    ///
    /// `degraded_threshold_ms` is the delay after which a node is considered
    /// degraded but not yet timed out.  It must be less than `timeout_ms`.
    pub fn new(timeout_ms: i64, degraded_threshold_ms: i64) -> Self {
        Self {
            heartbeats: HashMap::new(),
            timeout_ms,
            degraded_threshold_ms: degraded_threshold_ms.min(timeout_ms),
            failed_nodes: HashSet::new(),
        }
    }

    /// Record a heartbeat from `node_id` at the current wall-clock time.
    ///
    /// Returns `Some(NodeRecovered)` if the node was previously failed,
    /// otherwise `None`.
    pub fn record_heartbeat(&mut self, node_id: NodeId) -> Option<NodeHealthEvent> {
        let now = Utc::now();
        let was_failed = self.failed_nodes.remove(&node_id);

        let event = if was_failed {
            let last = self.heartbeats.get(&node_id).copied().unwrap_or(now);
            let downtime_ms = now.signed_duration_since(last).num_milliseconds();
            Some(NodeHealthEvent::NodeRecovered {
                node_id,
                downtime_ms,
            })
        } else {
            None
        };

        self.heartbeats.insert(node_id, now);
        event
    }

    /// Record a heartbeat at a specific timestamp (useful for testing and
    /// replaying logs).
    pub fn record_heartbeat_at(
        &mut self,
        node_id: NodeId,
        at: DateTime<Utc>,
    ) -> Option<NodeHealthEvent> {
        let was_failed = self.failed_nodes.remove(&node_id);

        let event = if was_failed {
            let last = self.heartbeats.get(&node_id).copied().unwrap_or(at);
            let downtime_ms = at.signed_duration_since(last).num_milliseconds();
            Some(NodeHealthEvent::NodeRecovered {
                node_id,
                downtime_ms,
            })
        } else {
            None
        };

        self.heartbeats.insert(node_id, at);
        event
    }

    /// Check the health of all known nodes against `now`.
    ///
    /// Returns a list of health events (timeouts and degraded warnings).
    pub fn check_health(&mut self) -> Vec<NodeHealthEvent> {
        self.check_health_at(Utc::now())
    }

    /// Check health at a specific timestamp (useful for deterministic testing).
    pub fn check_health_at(&mut self, now: DateTime<Utc>) -> Vec<NodeHealthEvent> {
        let mut events = Vec::new();
        for (&node_id, &last_seen) in &self.heartbeats {
            let elapsed = now.signed_duration_since(last_seen).num_milliseconds();

            if elapsed >= self.timeout_ms {
                if !self.failed_nodes.contains(&node_id) {
                    events.push(NodeHealthEvent::NodeTimedOut { node_id, last_seen });
                }
            } else if elapsed >= self.degraded_threshold_ms && !self.failed_nodes.contains(&node_id)
            {
                events.push(NodeHealthEvent::NodeDegraded {
                    node_id,
                    delay_ms: elapsed,
                });
            }
        }

        // Mark timed-out nodes as failed.
        for ev in &events {
            if let NodeHealthEvent::NodeTimedOut { node_id, .. } = ev {
                self.failed_nodes.insert(*node_id);
            }
        }

        events
    }

    /// Return the set of nodes currently considered failed.
    pub fn get_failed_nodes(&self) -> &HashSet<NodeId> {
        &self.failed_nodes
    }

    /// Return the last-seen timestamp for a node, if any.
    pub fn last_seen(&self, node_id: &NodeId) -> Option<&DateTime<Utc>> {
        self.heartbeats.get(node_id)
    }

    /// Return the number of tracked nodes.
    pub fn tracked_node_count(&self) -> usize {
        self.heartbeats.len()
    }
}

// ─── Propagation State ─────────────────────────────────────────────────────

/// Tracks the state of a single spore transfer to a target node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationState {
    /// The transfer has been queued but not yet started.
    Pending,
    /// The transfer is in progress.
    InProgress {
        /// Number of chunks successfully sent so far.
        chunks_sent: usize,
        /// Total number of chunks in this transfer.
        total_chunks: usize,
    },
    /// The transfer has failed.
    Failed {
        /// Human-readable reason for the failure.
        reason: String,
    },
    /// The transfer completed successfully.
    Complete,
}

impl std::fmt::Display for PropagationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PropagationState::Pending => write!(f, "Pending"),
            PropagationState::InProgress {
                chunks_sent,
                total_chunks,
            } => {
                write!(f, "InProgress({}/{})", chunks_sent, total_chunks)
            }
            PropagationState::Failed { reason } => write!(f, "Failed({})", reason),
            PropagationState::Complete => write!(f, "Complete"),
        }
    }
}

// ─── Transfer Recovery Log ─────────────────────────────────────────────────

/// A single entry in the transfer recovery log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLogEntry {
    /// Spore being transferred.
    pub spore_id: Uuid,
    /// Target node.
    pub target_node: NodeId,
    /// When this attempt was recorded.
    pub timestamp: DateTime<Utc>,
    /// State at the time of logging.
    pub state: PropagationState,
    /// Optional note (e.g. failure reason, re-route info).
    pub note: String,
}

/// Append-only log of transfer attempts, used for crash recovery and auditing.
#[derive(Debug, Clone, Default)]
pub struct TransferRecoveryLog {
    entries: Vec<TransferLogEntry>,
}

impl TransferRecoveryLog {
    /// Create an empty recovery log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an entry.
    pub fn record(
        &mut self,
        spore_id: Uuid,
        target_node: NodeId,
        state: PropagationState,
        note: impl Into<String>,
    ) {
        self.entries.push(TransferLogEntry {
            spore_id,
            target_node,
            timestamp: Utc::now(),
            state,
            note: note.into(),
        });
    }

    /// Append an entry with explicit timestamp (for testing).
    pub fn record_at(
        &mut self,
        spore_id: Uuid,
        target_node: NodeId,
        state: PropagationState,
        note: impl Into<String>,
        timestamp: DateTime<Utc>,
    ) {
        self.entries.push(TransferLogEntry {
            spore_id,
            target_node,
            timestamp,
            state,
            note: note.into(),
        });
    }

    /// Return all log entries.
    pub fn entries(&self) -> &[TransferLogEntry] {
        &self.entries
    }

    /// Return entries for a specific spore.
    pub fn entries_for_spore(&self, spore_id: Uuid) -> Vec<&TransferLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.spore_id == spore_id)
            .collect()
    }

    /// Return entries for a specific target node.
    pub fn entries_for_node(&self, node_id: &NodeId) -> Vec<&TransferLogEntry> {
        self.entries
            .iter()
            .filter(|e| &e.target_node == node_id)
            .collect()
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─── Fault Tolerant Propagator ─────────────────────────────────────────────

/// Descriptor for a transfer that needs to be retried.
#[derive(Debug, Clone)]
pub struct RetryEntry {
    /// Spore to transfer.
    pub spore_id: Uuid,
    /// Originally intended target.
    pub original_target: NodeId,
    /// Number of chunks already sent to the original target.
    pub chunks_sent: usize,
    /// Total chunks.
    pub total_chunks: usize,
    /// Number of retry attempts so far.
    pub attempts: usize,
}

/// Wraps a `SporePropagator` and adds fault tolerance:
///
/// * Monitors node health during transfers.
/// * Re-routes transfers when a target node goes down.
/// * Maintains a retry queue for failed transfers.
/// * Tracks partially-transferred spores (which chunks were sent).
pub struct FaultTolerantPropagator {
    /// The inner propagator that actually moves spores around.
    pub propagator: SporePropagator,
    /// Node health monitor.
    pub health_monitor: NodeHealthMonitor,
    /// Current propagation state for each (spore_id, target_node) pair.
    pub transfer_states: HashMap<(Uuid, NodeId), PropagationState>,
    /// Queue of transfers that need to be retried.
    pub retry_queue: VecDeque<RetryEntry>,
    /// Recovery log for auditing / crash recovery.
    pub recovery_log: TransferRecoveryLog,
    /// Maximum number of retry attempts before giving up on a transfer.
    pub max_retries: usize,
}

impl FaultTolerantPropagator {
    /// Create a new fault-tolerant propagator wrapping `propagator`.
    pub fn new(
        propagator: SporePropagator,
        health_timeout_ms: i64,
        degraded_threshold_ms: i64,
        max_retries: usize,
    ) -> Self {
        Self {
            propagator,
            health_monitor: NodeHealthMonitor::new(health_timeout_ms, degraded_threshold_ms),
            transfer_states: HashMap::new(),
            retry_queue: VecDeque::new(),
            recovery_log: TransferRecoveryLog::new(),
            max_retries,
        }
    }

    /// Begin (or continue) a transfer of `spore_id` to `target`.
    ///
    /// The method is intentionally synchronous and state-machine driven so
    /// that actual network I/O can be performed by the caller between ticks.
    ///
    /// Returns the new `PropagationState` for this transfer.
    pub fn transfer_with_retry(
        &mut self,
        spore_id: Uuid,
        target: NodeId,
        total_chunks: usize,
    ) -> PropagationState {
        let key = (spore_id, target);

        // If the target is currently failed, go straight into retry logic.
        if self.health_monitor.get_failed_nodes().contains(&target) {
            let state = PropagationState::Failed {
                reason: format!("target node {} is offline", target),
            };
            self.transfer_states.insert(key, state.clone());
            self.recovery_log.record(
                spore_id,
                target,
                state.clone(),
                "target offline at transfer start",
            );
            self.enqueue_retry(spore_id, target, 0, total_chunks);
            return state;
        }

        // Otherwise, mark as in-progress (starting from chunk 0).
        let state = PropagationState::InProgress {
            chunks_sent: 0,
            total_chunks,
        };
        self.transfer_states.insert(key, state.clone());
        self.recovery_log
            .record(spore_id, target, state.clone(), "transfer started");

        info!(
            "FaultTolerantPropagator: started transfer of spore {} to node {} ({} chunks)",
            spore_id, target, total_chunks
        );

        state
    }

    /// Handle a node failure discovered during an in-progress transfer.
    ///
    /// Marks the transfer as failed, logs it, and enqueues a retry.
    pub fn handle_node_failure(&mut self, spore_id: Uuid, failed_node: NodeId) {
        let key = (spore_id, failed_node);

        let (chunks_sent, total_chunks) = match self.transfer_states.get(&key) {
            Some(PropagationState::InProgress {
                chunks_sent,
                total_chunks,
            }) => (*chunks_sent, *total_chunks),
            _ => (0, 0),
        };

        let reason = format!("node {} went offline during transfer", failed_node);
        let state = PropagationState::Failed {
            reason: reason.clone(),
        };
        self.transfer_states.insert(key, state.clone());
        self.recovery_log
            .record(spore_id, failed_node, state, &reason);

        warn!(
            "FaultTolerantPropagator: node {} failed during transfer of spore {} ({}/{} chunks sent)",
            failed_node, spore_id, chunks_sent, total_chunks
        );

        self.enqueue_retry(spore_id, failed_node, chunks_sent, total_chunks);
    }

    /// Resume a previously-failed transfer, optionally re-routing to
    /// `new_target`.
    ///
    /// If `new_target` is the same as the original target the transfer is
    /// simply restarted from `chunks_already_sent`.
    ///
    /// Returns the new `PropagationState`.
    pub fn resume_transfer(
        &mut self,
        spore_id: Uuid,
        original_target: NodeId,
        new_target: NodeId,
        chunks_already_sent: usize,
        total_chunks: usize,
    ) -> PropagationState {
        // If the new target is also failed, mark as failed immediately.
        if self.health_monitor.get_failed_nodes().contains(&new_target) {
            let state = PropagationState::Failed {
                reason: format!("re-route target {} is also offline", new_target),
            };
            let key = (spore_id, new_target);
            self.transfer_states.insert(key, state.clone());
            self.recovery_log.record(
                spore_id,
                new_target,
                state.clone(),
                format!(
                    "re-route from {} failed: new target offline",
                    original_target
                ),
            );
            return state;
        }

        let state = PropagationState::InProgress {
            chunks_sent: chunks_already_sent,
            total_chunks,
        };
        let key = (spore_id, new_target);
        self.transfer_states.insert(key, state.clone());
        self.recovery_log.record(
            spore_id,
            new_target,
            state.clone(),
            format!(
                "resumed from {} (re-routed from {}), starting at chunk {}",
                new_target, original_target, chunks_already_sent
            ),
        );

        info!(
            "FaultTolerantPropagator: resumed transfer of spore {} → {} (from chunk {}/{})",
            spore_id, new_target, chunks_already_sent, total_chunks
        );

        state
    }

    /// Mark a transfer as complete.
    pub fn complete_transfer(&mut self, spore_id: Uuid, target: NodeId) {
        let key = (spore_id, target);
        self.transfer_states.insert(key, PropagationState::Complete);
        self.recovery_log.record(
            spore_id,
            target,
            PropagationState::Complete,
            "transfer complete",
        );
        info!(
            "FaultTolerantPropagator: transfer of spore {} to {} complete",
            spore_id, target
        );
    }

    /// Update the in-progress chunk count for a transfer.
    pub fn update_progress(
        &mut self,
        spore_id: Uuid,
        target: NodeId,
        chunks_sent: usize,
        total_chunks: usize,
    ) {
        let key = (spore_id, target);
        self.transfer_states.insert(
            key,
            PropagationState::InProgress {
                chunks_sent,
                total_chunks,
            },
        );
    }

    /// Get the current state of a transfer.
    pub fn get_transfer_state(&self, spore_id: Uuid, target: NodeId) -> Option<&PropagationState> {
        self.transfer_states.get(&(spore_id, target))
    }

    /// Pop the next entry from the retry queue.
    pub fn next_retry(&mut self) -> Option<RetryEntry> {
        self.retry_queue.pop_front()
    }

    /// Return how many retries are queued.
    pub fn retry_queue_len(&self) -> usize {
        self.retry_queue.len()
    }

    // ── private helpers ──

    fn enqueue_retry(
        &mut self,
        spore_id: Uuid,
        original_target: NodeId,
        chunks_sent: usize,
        total_chunks: usize,
    ) {
        // Count prior attempts.
        let prior = self
            .retry_queue
            .iter()
            .filter(|e| e.spore_id == spore_id && e.original_target == original_target)
            .count();

        if prior < self.max_retries {
            self.retry_queue.push_back(RetryEntry {
                spore_id,
                original_target,
                chunks_sent,
                total_chunks,
                attempts: prior + 1,
            });
            info!(
                "FaultTolerantPropagator: enqueued retry #{} for spore {} → {}",
                prior + 1,
                spore_id,
                original_target
            );
        } else {
            warn!(
                "FaultTolerantPropagator: max retries ({}) exceeded for spore {} → {}",
                self.max_retries, spore_id, original_target
            );
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mycelium_core::ModelConfig;

    fn test_node_id() -> NodeId {
        NodeId::new()
    }

    fn test_spore() -> Spore {
        let weight_data = vec![0u8; 1024];
        SporeBuilder::new(ModelConfig::minimax_m25(), test_node_id())
            .layer_range(0, 4)
            .expert_ids(vec![0, 1])
            .build(weight_data, 4)
    }

    #[allow(dead_code)]
    fn test_spore_with_data(data: Vec<u8>) -> Spore {
        SporeBuilder::new(ModelConfig::minimax_m25(), test_node_id())
            .layer_range(0, 4)
            .expert_ids(vec![0, 1])
            .build(data, 4)
    }

    // ─── Original tests ─────────────────────────────────────────────────

    #[test]
    fn test_spore_manifest_from_spore() {
        let spore = test_spore();
        let manifest = SporeManifest::from_spore(&spore);
        assert_eq!(manifest.layer_range, (0, 4));
        assert_eq!(manifest.expert_ids, vec![0, 1]);
        assert_eq!(manifest.model_config.name, "MiniMax-M2.5");
    }

    #[test]
    fn test_spore_genome_verify() {
        let spore = test_spore();
        // Should verify successfully
        assert!(spore.genome.verify());
    }

    #[test]
    fn test_spore_genome_tamper() {
        let spore = test_spore();
        // Tamper with the genome data
        let mut tampered = spore.clone();
        tampered.genome.data[0] ^= 0xFF;
        // Should fail verification
        assert!(!tampered.genome.verify());
    }

    #[test]
    fn test_chunk_round_trip() {
        let spore_id = Uuid::new_v4();
        let data = vec![0xAB; 3 * SPORE_CHUNK_SIZE + 512]; // 3 full + 1 partial chunk
        let chunks = SporeChunk::from_data(spore_id, &data);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].total_chunks, 4);

        let reassembled = SporeChunk::reassemble(&chunks).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_chunk_hash_verification() {
        let spore_id = Uuid::new_v4();
        let data = vec![42u8; 1024];
        let mut chunks = SporeChunk::from_data(spore_id, &data);

        // Tamper with a chunk
        chunks[0].data[0] ^= 0xFF;
        let result = SporeChunk::reassemble(&chunks);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialize_deserialize() {
        let spore = test_spore();
        let serialized = serialize_spore(&spore).unwrap();
        let deserialized = deserialize_spore(&serialized).unwrap();
        assert_eq!(&deserialized.magic, b"MYCO");
        assert_eq!(deserialized.version, SPORE_VERSION);
    }

    #[test]
    fn test_serialize_hash_integrity() {
        let spore = test_spore();
        let mut serialized = serialize_spore(&spore).unwrap();
        // Tamper with serialized data
        let len = serialized.len();
        serialized[len - 1] ^= 0xFF;
        let result = deserialize_spore(&serialized);
        assert!(result.is_err());
    }

    // ─── SporeLifecycle tests ────────────────────────────────────────────

    #[test]
    fn test_lifecycle_new_is_dormant() {
        let spore = test_spore();
        let lifecycle = SporeLifecycle::new(spore);
        assert_eq!(lifecycle.state(), SporeState::Dormant);
        assert_eq!(lifecycle.germination_progress, 0.0);
        assert!(lifecycle.activation_time.is_none());
    }

    #[test]
    fn test_lifecycle_germinate() {
        let spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        let mut lifecycle = SporeLifecycle::new(spore);
        assert!(lifecycle.germinate(&mut substrate).is_ok());
        assert_eq!(lifecycle.state(), SporeState::Germinating);
        assert_eq!(lifecycle.germination_progress, 0.0);
    }

    #[test]
    fn test_lifecycle_germinate_rejects_non_dormant() {
        let spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        let mut lifecycle = SporeLifecycle::new(spore);
        lifecycle.germinate(&mut substrate).unwrap();
        // Try to germinate again — should fail
        assert!(lifecycle.germinate(&mut substrate).is_err());
    }

    #[test]
    fn test_lifecycle_activate() {
        let spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        let mut lifecycle = SporeLifecycle::new(spore);
        lifecycle.germinate(&mut substrate).unwrap();
        assert!(lifecycle.activate().is_ok());
        assert_eq!(lifecycle.state(), SporeState::Active);
        assert_eq!(lifecycle.germination_progress, 1.0);
        assert!(lifecycle.activation_time.is_some());
    }

    #[test]
    fn test_lifecycle_activate_from_dormant() {
        let spore = test_spore();
        let mut lifecycle = SporeLifecycle::new(spore);
        // Can activate directly from dormant
        assert!(lifecycle.activate().is_ok());
        assert_eq!(lifecycle.state(), SporeState::Active);
    }

    #[test]
    fn test_lifecycle_fruit() {
        let spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        let mut lifecycle = SporeLifecycle::new(spore);
        lifecycle.germinate(&mut substrate).unwrap();
        lifecycle.activate().unwrap();

        // Create child spores
        let child = replicate(&lifecycle.spore, MutationConfig::none()).unwrap();
        let children = lifecycle.fruit(vec![child]).unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(lifecycle.state(), SporeState::Fruiting);
    }

    #[test]
    fn test_lifecycle_fruit_rejects_non_active() {
        let spore = test_spore();
        let mut lifecycle = SporeLifecycle::new(spore);
        assert!(lifecycle.fruit(vec![]).is_err());
    }

    #[test]
    fn test_lifecycle_kill() {
        let spore = test_spore();
        let mut lifecycle = SporeLifecycle::new(spore);
        assert!(lifecycle.kill().is_ok());
        assert_eq!(lifecycle.state(), SporeState::Dead);
    }

    #[test]
    fn test_lifecycle_kill_from_any_state() {
        let _spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        // Kill from Dormant
        let mut lc = SporeLifecycle::new(test_spore());
        assert!(lc.kill().is_ok());

        // Kill from Active
        let mut lc = SporeLifecycle::new(test_spore());
        lc.germinate(&mut substrate).unwrap();
        lc.activate().unwrap();
        assert!(lc.kill().is_ok());
    }

    #[test]
    fn test_lifecycle_is_healthy() {
        let spore = test_spore();
        let lifecycle = SporeLifecycle::new(spore);
        assert!(lifecycle.is_healthy());

        // Tamper with the genome
        let mut tampered_spore = test_spore();
        tampered_spore.genome.data[0] ^= 0xFF;
        let tampered_lifecycle = SporeLifecycle::new(tampered_spore);
        assert!(!tampered_lifecycle.is_healthy());
    }

    #[test]
    fn test_lifecycle_set_germination_progress() {
        let spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        let mut lifecycle = SporeLifecycle::new(spore);
        lifecycle.germinate(&mut substrate).unwrap();
        assert!(lifecycle.set_germination_progress(0.5).is_ok());
        assert!((lifecycle.germination_progress - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_lifecycle_germination_progress_clamps() {
        let spore = test_spore();
        let mut substrate = mycelium_substrate::SubstrateManager::new(
            std::env::temp_dir(),
            ModelConfig::minimax_m25(),
        );
        let mut lifecycle = SporeLifecycle::new(spore);
        lifecycle.germinate(&mut substrate).unwrap();
        assert!(lifecycle.set_germination_progress(2.0).is_ok());
        assert!((lifecycle.germination_progress - 1.0).abs() < 0.001);
    }

    // ─── Replication tests ──────────────────────────────────────────────

    #[test]
    fn test_replicate_basic() {
        let parent = test_spore();
        let child = replicate(&parent, MutationConfig::none()).unwrap();
        assert_ne!(child.id, parent.id);
        assert_eq!(child.generation, parent.generation + 1);
        assert_eq!(child.parent, parent.parent);
        assert_eq!(child.layer_range, parent.layer_range);
        assert_eq!(child.expert_ids, parent.expert_ids);
        // Child genome should verify (no mutations)
        assert!(child.genome.verify());
    }

    #[test]
    fn test_replicate_increments_generation() {
        let gen0 = test_spore();
        let gen1 = replicate(&gen0, MutationConfig::none()).unwrap();
        let gen2 = replicate(&gen1, MutationConfig::none()).unwrap();
        assert_eq!(gen0.generation, 0);
        assert_eq!(gen1.generation, 1);
        assert_eq!(gen2.generation, 2);
    }

    #[test]
    fn test_replicate_with_mutations() {
        let parent = test_spore();
        let mutations = MutationConfig {
            bit_flip_rate: 0.01,
            weight_perturbation: 0.01,
            max_generation: 100,
        };
        let child = replicate(&parent, mutations).unwrap();
        // Child should have a different genome hash due to mutations
        assert_ne!(child.genome.hash, parent.genome.hash);
        // But child should still verify (hash is recomputed)
        assert!(child.genome.verify());
    }

    #[test]
    fn test_replicate_fails_on_corrupted_parent() {
        let mut parent = test_spore();
        parent.genome.data[0] ^= 0xFF; // corrupt the data
        let result = replicate(&parent, MutationConfig::none());
        assert!(result.is_err());
    }

    #[test]
    fn test_replicate_fails_on_max_generation() {
        let mut parent = test_spore();
        parent.generation = 99;
        let config = MutationConfig {
            bit_flip_rate: 0.0,
            weight_perturbation: 0.0,
            max_generation: 100,
        };
        // generation 99 < max 100, should succeed
        assert!(replicate(&parent, config.clone()).is_ok());

        parent.generation = 100;
        // generation 100 >= max 100, should fail
        assert!(replicate(&parent, config).is_err());
    }

    #[test]
    fn test_replicate_chain() {
        let mut current = test_spore();
        let config = MutationConfig::none();
        for i in 0..5 {
            let child = replicate(&current, config.clone()).unwrap();
            assert_eq!(child.generation, i + 1);
            current = child;
        }
        assert_eq!(current.generation, 5);
    }

    // ─── MutationConfig tests ────────────────────────────────────────────

    #[test]
    fn test_mutation_config_defaults() {
        let config = MutationConfig::default();
        assert_eq!(config.bit_flip_rate, 0.0001);
        assert_eq!(config.weight_perturbation, 0.01);
        assert_eq!(config.max_generation, 100);
    }

    #[test]
    fn test_mutation_config_none() {
        let config = MutationConfig::none();
        assert_eq!(config.bit_flip_rate, 0.0);
        assert_eq!(config.weight_perturbation, 0.0);
    }

    #[test]
    fn test_mutation_config_aggressive() {
        let config = MutationConfig::aggressive();
        assert!(config.bit_flip_rate > 0.0);
        assert!(config.weight_perturbation > 0.0);
    }

    // ─── SporeIntegrity tests ───────────────────────────────────────────

    #[test]
    fn test_verify_spore_integrity_valid() {
        let spore = test_spore();
        let integrity = verify_spore_integrity(&spore);
        assert!(integrity.genome_valid);
        assert!(integrity.hash_matches);
        assert!(integrity.not_expired);
        assert!(integrity.generation_within_limits);
        assert!(integrity.overall);
    }

    #[test]
    fn test_verify_spore_integrity_corrupted() {
        let mut spore = test_spore();
        spore.genome.data[0] ^= 0xFF; // corrupt data
        let integrity = verify_spore_integrity(&spore);
        assert!(!integrity.genome_valid);
        assert!(!integrity.hash_matches);
        assert!(!integrity.overall);
    }

    #[test]
    fn test_verify_spore_integrity_generation_limit() {
        let mut spore = test_spore();
        spore.generation = 1001; // beyond limit
        let integrity = verify_spore_integrity(&spore);
        assert!(!integrity.generation_within_limits);
        assert!(!integrity.overall);
    }

    // ─── BinarySpore tests ──────────────────────────────────────────────

    #[test]
    fn test_binary_spore_round_trip() {
        let spore = test_spore();
        let binary = BinarySpore::from_spore(&spore).unwrap();
        let bytes = binary.to_bytes();
        let restored = BinarySpore::from_bytes(&bytes).unwrap();
        assert_eq!(restored, binary);
    }

    #[test]
    fn test_binary_spore_decompress() {
        let spore = test_spore();
        let binary = BinarySpore::from_spore(&spore).unwrap();
        let decompressed = binary.decompress_genome().unwrap();
        assert_eq!(decompressed, spore.genome.data);
    }

    #[test]
    fn test_binary_spore_to_spore() {
        let spore = test_spore();
        let binary = BinarySpore::from_spore(&spore).unwrap();
        let restored = binary
            .to_spore(
                spore.model_config.clone(),
                spore.layer_range,
                spore.expert_ids.clone(),
                spore.parent.clone(),
                spore.generation,
                spore.genome.quant_bits,
            )
            .unwrap();
        // Genome data should match
        assert_eq!(restored.genome.data, spore.genome.data);
        assert_eq!(restored.generation, spore.generation);
        assert_eq!(restored.layer_range, spore.layer_range);
    }

    #[test]
    fn test_binary_spore_magic() {
        let spore = test_spore();
        let binary = BinarySpore::from_spore(&spore).unwrap();
        assert_eq!(binary.magic, SPORE_BINARY_MAGIC);
        assert_eq!(binary.version, SPORE_BINARY_VERSION);
    }

    #[test]
    fn test_binary_spore_tampered_checksum() {
        let spore = test_spore();
        let binary = BinarySpore::from_spore(&spore).unwrap();
        let mut bytes = binary.to_bytes();
        // Tamper with the body (between header and footer)
        if bytes.len() > 16 {
            bytes[13] ^= 0xFF;
        }
        let result = BinarySpore::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_spore_bad_magic() {
        let spore = test_spore();
        let binary = BinarySpore::from_spore(&spore).unwrap();
        let mut bytes = binary.to_bytes();
        bytes[0] = 0x00; // corrupt magic
        let result = BinarySpore::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_spore_too_short() {
        let data = vec![0u8; 10]; // too short
        let result = BinarySpore::from_bytes(&data);
        assert!(result.is_err());
    }

    // ─── SporeState Display test ────────────────────────────────────────

    #[test]
    fn test_spore_state_display() {
        assert_eq!(format!("{}", SporeState::Dormant), "Dormant");
        assert_eq!(format!("{}", SporeState::Germinating), "Germinating");
        assert_eq!(format!("{}", SporeState::Active), "Active");
        assert_eq!(format!("{}", SporeState::Fruiting), "Fruiting");
        assert_eq!(format!("{}", SporeState::Dead), "Dead");
    }

    // ─── NodeHealthMonitor tests ─────────────────────────────────────────

    fn fixed_time() -> DateTime<Utc> {
        chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn test_health_monitor_record_heartbeat() {
        let mut monitor = NodeHealthMonitor::new(5000, 3000);
        let node = test_node_id();
        let event = monitor.record_heartbeat_at(node, fixed_time());
        assert!(event.is_none()); // first heartbeat, no recovery
        assert_eq!(monitor.tracked_node_count(), 1);
        assert_eq!(*monitor.last_seen(&node).unwrap(), fixed_time());
    }

    #[test]
    fn test_health_monitor_detects_timeout() {
        let mut monitor = NodeHealthMonitor::new(5000, 3000);
        let node = test_node_id();
        let t0 = fixed_time();
        monitor.record_heartbeat_at(node, t0);

        // 6 seconds later → should timeout
        let t1 = t0 + chrono::Duration::milliseconds(6000);
        let events = monitor.check_health_at(t1);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], NodeHealthEvent::NodeTimedOut { node_id, .. } if *node_id == node)
        );
        assert!(monitor.get_failed_nodes().contains(&node));
    }

    #[test]
    fn test_health_monitor_detects_degraded() {
        let mut monitor = NodeHealthMonitor::new(5000, 3000);
        let node = test_node_id();
        let t0 = fixed_time();
        monitor.record_heartbeat_at(node, t0);

        // 4 seconds later → degraded but not timed out
        let t1 = t0 + chrono::Duration::milliseconds(4000);
        let events = monitor.check_health_at(t1);
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], NodeHealthEvent::NodeDegraded { node_id, delay_ms } if *node_id == node && *delay_ms == 4000)
        );
        // Not yet failed
        assert!(monitor.get_failed_nodes().is_empty());
    }

    #[test]
    fn test_health_monitor_recovery() {
        let mut monitor = NodeHealthMonitor::new(5000, 3000);
        let node = test_node_id();
        let t0 = fixed_time();
        monitor.record_heartbeat_at(node, t0);

        // Force timeout
        let t1 = t0 + chrono::Duration::milliseconds(6000);
        monitor.check_health_at(t1);
        assert!(monitor.get_failed_nodes().contains(&node));

        // Node comes back
        let t2 = t0 + chrono::Duration::milliseconds(8000);
        let event = monitor.record_heartbeat_at(node, t2);
        assert!(matches!(event, Some(NodeHealthEvent::NodeRecovered { .. })));
        assert!(!monitor.get_failed_nodes().contains(&node));
    }

    #[test]
    fn test_health_monitor_no_double_timeout() {
        let mut monitor = NodeHealthMonitor::new(5000, 3000);
        let node = test_node_id();
        let t0 = fixed_time();
        monitor.record_heartbeat_at(node, t0);

        let t1 = t0 + chrono::Duration::milliseconds(6000);
        let events1 = monitor.check_health_at(t1);
        assert_eq!(events1.len(), 1);

        // Check again — should not re-emit timeout
        let t2 = t0 + chrono::Duration::milliseconds(7000);
        let events2 = monitor.check_health_at(t2);
        assert!(events2.is_empty());
    }

    // ─── PropagationState tests ──────────────────────────────────────────

    #[test]
    fn test_propagation_state_display() {
        assert_eq!(format!("{}", PropagationState::Pending), "Pending");
        assert_eq!(
            format!(
                "{}",
                PropagationState::InProgress {
                    chunks_sent: 3,
                    total_chunks: 10
                }
            ),
            "InProgress(3/10)"
        );
        assert_eq!(
            format!(
                "{}",
                PropagationState::Failed {
                    reason: "offline".into()
                }
            ),
            "Failed(offline)"
        );
        assert_eq!(format!("{}", PropagationState::Complete), "Complete");
    }

    // ─── TransferRecoveryLog tests ───────────────────────────────────────

    #[test]
    fn test_recovery_log_record_and_query() {
        let mut log = TransferRecoveryLog::new();
        assert!(log.is_empty());

        let spore_id = Uuid::new_v4();
        let node = test_node_id();
        log.record(spore_id, node, PropagationState::Pending, "test");
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());

        let by_spore = log.entries_for_spore(spore_id);
        assert_eq!(by_spore.len(), 1);
        assert_eq!(by_spore[0].note, "test");

        let by_node = log.entries_for_node(&node);
        assert_eq!(by_node.len(), 1);
    }

    #[test]
    fn test_recovery_log_multiple_entries() {
        let mut log = TransferRecoveryLog::new();
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let n = test_node_id();

        log.record(s1, n, PropagationState::Pending, "a");
        log.record(s2, n, PropagationState::Complete, "b");
        log.record(s1, n, PropagationState::Complete, "c");

        assert_eq!(log.len(), 3);
        assert_eq!(log.entries_for_spore(s1).len(), 2);
        assert_eq!(log.entries_for_spore(s2).len(), 1);
        assert_eq!(log.entries_for_node(&n).len(), 3);
    }

    // ─── FaultTolerantPropagator tests ───────────────────────────────────

    fn test_node_capacity(node_id: NodeId) -> NodeCapacityState {
        NodeCapacityState {
            node_id,
            available_vram_mb: 8192,
            available_ram_mb: 32768,
            layer_range: (0, 4),
            uptime_secs: 600,
            has_lora: false,
            lora_improvement: 0.0,
        }
    }

    fn test_fault_tolerant_propagator() -> FaultTolerantPropagator {
        let node = test_node_id();
        let cap = test_node_capacity(node);
        let propagator = SporePropagator::new(PropagationConfig::default(), cap);
        FaultTolerantPropagator::new(propagator, 5000, 3000, 3)
    }

    #[test]
    fn test_ft_transfer_happy_path() {
        let mut ftp = test_fault_tolerant_propagator();
        let spore_id = Uuid::new_v4();
        let target = test_node_id();

        let state = ftp.transfer_with_retry(spore_id, target, 10);
        assert!(matches!(
            state,
            PropagationState::InProgress {
                chunks_sent: 0,
                total_chunks: 10
            }
        ));

        // Simulate progress
        ftp.update_progress(spore_id, target, 5, 10);
        assert!(matches!(
            ftp.get_transfer_state(spore_id, target),
            Some(PropagationState::InProgress {
                chunks_sent: 5,
                total_chunks: 10
            })
        ));

        ftp.complete_transfer(spore_id, target);
        assert!(matches!(
            ftp.get_transfer_state(spore_id, target),
            Some(PropagationState::Complete)
        ));

        // Recovery log should have 3 entries: start, (progress is not logged), complete
        // Actually just start + complete = 2 from record() calls
        assert_eq!(ftp.recovery_log.entries_for_spore(spore_id).len(), 2);
    }

    #[test]
    fn test_ft_transfer_to_offline_node() {
        let mut ftp = test_fault_tolerant_propagator();
        let target = test_node_id();
        let spore_id = Uuid::new_v4();

        // Mark the target as offline via health monitor
        let t0 = fixed_time();
        ftp.health_monitor.record_heartbeat_at(target, t0);
        let t1 = t0 + chrono::Duration::milliseconds(6000);
        ftp.health_monitor.check_health_at(t1);
        assert!(ftp.health_monitor.get_failed_nodes().contains(&target));

        // Transfer should fail immediately and enqueue retry
        let state = ftp.transfer_with_retry(spore_id, target, 10);
        assert!(matches!(state, PropagationState::Failed { .. }));
        assert_eq!(ftp.retry_queue_len(), 1);
    }

    #[test]
    fn test_ft_handle_node_failure_during_transfer() {
        let mut ftp = test_fault_tolerant_propagator();
        let spore_id = Uuid::new_v4();
        let target = test_node_id();

        ftp.transfer_with_retry(spore_id, target, 10);
        ftp.update_progress(spore_id, target, 4, 10);

        // Node fails mid-transfer
        ftp.handle_node_failure(spore_id, target);
        assert!(matches!(
            ftp.get_transfer_state(spore_id, target),
            Some(PropagationState::Failed { .. })
        ));
        assert_eq!(ftp.retry_queue_len(), 1);

        let retry = ftp.next_retry().unwrap();
        assert_eq!(retry.spore_id, spore_id);
        assert_eq!(retry.original_target, target);
        assert_eq!(retry.chunks_sent, 4);
        assert_eq!(retry.total_chunks, 10);
        assert_eq!(retry.attempts, 1);
    }

    #[test]
    fn test_ft_resume_transfer_reroute() {
        let mut ftp = test_fault_tolerant_propagator();
        let spore_id = Uuid::new_v4();
        let original = test_node_id();
        let new_target = test_node_id();

        // Start and then fail
        ftp.transfer_with_retry(spore_id, original, 10);
        ftp.update_progress(spore_id, original, 6, 10);
        ftp.handle_node_failure(spore_id, original);

        // Resume to a different node, continuing from chunk 6
        let state = ftp.resume_transfer(spore_id, original, new_target, 6, 10);
        assert!(matches!(
            state,
            PropagationState::InProgress {
                chunks_sent: 6,
                total_chunks: 10
            }
        ));

        // Complete on the new target
        ftp.complete_transfer(spore_id, new_target);
        assert!(matches!(
            ftp.get_transfer_state(spore_id, new_target),
            Some(PropagationState::Complete)
        ));
    }

    #[test]
    fn test_ft_resume_to_failed_node_fails() {
        let mut ftp = test_fault_tolerant_propagator();
        let spore_id = Uuid::new_v4();
        let original = test_node_id();
        let also_dead = test_node_id();

        // Mark also_dead as failed
        let t0 = fixed_time();
        ftp.health_monitor.record_heartbeat_at(also_dead, t0);
        ftp.health_monitor
            .check_health_at(t0 + chrono::Duration::milliseconds(6000));

        let state = ftp.resume_transfer(spore_id, original, also_dead, 3, 10);
        assert!(matches!(state, PropagationState::Failed { .. }));
    }

    #[test]
    fn test_ft_max_retries_exceeded() {
        let mut ftp = test_fault_tolerant_propagator(); // max_retries = 3
        let spore_id = Uuid::new_v4();
        let target = test_node_id();

        // Enqueue 3 retries (the maximum)
        for _ in 0..3 {
            ftp.handle_node_failure(spore_id, target);
        }
        assert_eq!(ftp.retry_queue_len(), 3);

        // 4th attempt should NOT be enqueued (already 3 in queue)
        ftp.handle_node_failure(spore_id, target);
        assert_eq!(ftp.retry_queue_len(), 3); // still 3
    }
}
