//! # Mycelium Nucleus — Federated Self-Tuning Engine
//!
//! The nucleus is the learning core of each mycelium node.
//! It implements:
//! - Local LoRA fine-tuning on collected experience
//! - Federated gradient aggregation across the network
//! - Continuous self-improvement loop
//!
//! The LoRA adapter has shape:
//! - A: (rank × hidden_dim) — down-projection
//! - B: (hidden_dim × rank) — up-projection
//! - Output: ΔW = α * B @ A (applied per target layer)
//!
//! # Gradient Flow Architecture
//! ```text
//! [Inference Request] → Model Forward Pass → [Output + Latents]
//!                                                   ↓
//!                                           [Latent Collector]
//!                                                   ↓
//! [Reward Signal] ← User Feedback ←───────────────┘
//!       ↓
//! [Training Sample] → Nucleus.train_step() → [Gradient Deltas]
//!                                               ↓
//!                                    [Federated Averaging]
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use mycelium_core::{
    FEDAVG_MIN_PARTICIPANTS, HyphaeMessage, LORA_DEFAULT_RANK, LatentVector, LoRAAdapter,
    ModelConfig, NodeId,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};
use uuid::Uuid;

// ─── Training Sample (defined here, not in core) ───────────────────────────

/// A training sample for the self-tuning engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input latent vector
    pub input_latent: LatentVector,
    /// Target latent vector
    pub target_latent: LatentVector,
    /// Reward signal for this sample
    pub reward: f32,
    /// Where this sample came from
    pub source: SampleSource,
    /// When this sample was collected
    pub timestamp: DateTime<Utc>,
}

/// Source of a training sample.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SampleSource {
    /// Generated from self-play
    SelfPlay,
    /// Provided by user feedback
    UserFeedback,
    /// Generated from federated averaging
    Federated,
    /// Generated from exploration
    Exploration,
}

// ─── Training Config ────────────────────────────────────────────────────────

/// Configuration for nucleus training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// LoRA rank
    pub lora_rank: usize,
    /// LoRA alpha (scaling factor)
    pub lora_alpha: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Maximum experience buffer size
    pub max_buffer_size: usize,
    /// Gradient clipping max norm
    pub max_grad_norm: f64,
    /// Weight decay for AdamW
    pub weight_decay: f64,
    /// Minimum reward threshold for including in training
    pub min_reward: f32,
    /// Number of training steps between federated averaging rounds
    pub fedavg_interval: u64,
    /// Differential privacy noise scale (0 = disabled)
    pub dp_noise_scale: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            lora_rank: LORA_DEFAULT_RANK,
            lora_alpha: 16.0,
            batch_size: 8,
            max_buffer_size: 10000,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            min_reward: 0.3,
            fedavg_interval: 100,
            dp_noise_scale: 0.0,
        }
    }
}

// ─── Federated Averaging State ─────────────────────────────────────────────

/// State for federated averaging across the network.
#[derive(Debug, Clone, Default)]
pub struct FedAvgState {
    /// Accumulated gradient deltas from peers: node_id -> list of (layer_idx, delta, version)
    pub peer_deltas: HashMap<NodeId, Vec<(usize, Vec<f32>, u64)>>,
    /// Current global version
    pub global_version: u64,
    /// Number of participants in current round
    pub participants: usize,
    /// Whether we've received enough deltas for a round
    pub round_complete: bool,
}

impl FedAvgState {
    /// Add a peer's gradient delta.
    pub fn add_peer_delta(
        &mut self,
        layer_idx: usize,
        delta: Vec<f32>,
        version: u64,
        node_id: NodeId,
    ) {
        self.peer_deltas
            .entry(node_id)
            .or_default()
            .push((layer_idx, delta, version));
        self.participants = self.peer_deltas.len();
        self.round_complete = self.participants >= FEDAVG_MIN_PARTICIPANTS;
    }

    /// Process a GradientDelta HyphaeMessage.
    pub fn add_gradient_message(&mut self, msg: &HyphaeMessage) {
        if let HyphaeMessage::GradientDelta {
            layer_idx,
            delta,
            version,
            node_id,
        } = msg
        {
            self.add_peer_delta(*layer_idx, delta.clone(), *version, *node_id);
        }
    }

    /// Perform federated averaging if round is complete.
    pub fn federated_average(&mut self) -> Option<Vec<(usize, Vec<f32>)>> {
        if !self.round_complete {
            return None;
        }

        let mut averaged: Vec<(usize, Vec<f32>)> = Vec::new();
        let num_peers = self.peer_deltas.len() as f32;

        // Group deltas by layer
        let mut by_layer: HashMap<usize, Vec<&Vec<f32>>> = HashMap::new();
        for deltas in self.peer_deltas.values() {
            for (layer_idx, delta, _version) in deltas {
                by_layer.entry(*layer_idx).or_default().push(delta);
            }
        }

        // Average each layer's deltas
        for (layer_idx, deltas) in by_layer {
            if deltas.is_empty() {
                continue;
            }

            let len = deltas[0].len();
            let mut avg = vec![0.0f32; len];

            for delta in &deltas {
                for (i, &v) in delta.iter().enumerate() {
                    if i < len {
                        avg[i] += v / num_peers;
                    }
                }
            }

            averaged.push((layer_idx, avg));
        }

        self.global_version += 1;
        self.peer_deltas.clear();
        self.participants = 0;
        self.round_complete = false;

        info!(
            "Federated averaging complete: version {}, {} layers",
            self.global_version,
            averaged.len()
        );

        Some(averaged)
    }
}

// ─── Nucleus (Self-Tuning Engine) ─────────────────────────────────────────

/// The nucleus — the self-tuning engine of each mycelium node.
///
/// Manages a LoRA adapter that is fine-tuned on collected experience
/// and shares gradient updates with the network via federated averaging.
///
/// LoRA math: ΔW = α/rank * B @ A
/// - A: (rank × hidden_dim) — down-projection, initialized with Kaiming
/// - B: (hidden_dim × rank) — up-projection, initialized at zero
pub struct Nucleus {
    /// Current LoRA adapter weights
    adapter: LoRAAdapter,
    /// Collected training samples
    experience_buffer: Vec<TrainingSample>,
    /// Federated averaging state
    fedavg: FedAvgState,
    /// Model config
    config: ModelConfig,
    /// Training config
    train_config: TrainingConfig,
    /// Node ID
    node_id: NodeId,
    /// Training step counter
    step: u64,
    /// Running loss average for monitoring
    running_loss: f64,
    /// Best loss seen (for checkpointing)
    best_loss: f64,
    /// AdamW first moment (m) for A weights — one per row
    m_a: Vec<Vec<f32>>,
    /// AdamW second moment (v) for A weights
    v_a: Vec<Vec<f32>>,
    /// AdamW first moment for B weights
    m_b: Vec<Vec<f32>>,
    /// AdamW second moment for B weights
    v_b: Vec<Vec<f32>>,
}

impl Nucleus {
    /// Create a new nucleus with default training config.
    pub fn new(config: ModelConfig, node_id: NodeId) -> Self {
        Self::with_train_config(config, node_id, TrainingConfig::default())
    }

    /// Create a new nucleus with custom training config.
    pub fn with_train_config(
        config: ModelConfig,
        node_id: NodeId,
        train_config: TrainingConfig,
    ) -> Self {
        let rank = train_config.lora_rank;
        let hidden_dim = config.hidden_dim;

        // Initialize LoRA adapter
        // A: (rank × hidden_dim) — Kaiming init
        let a_weights: Vec<Vec<f32>> = (0..rank)
            .map(|_| {
                let scale = (2.0 / rank as f64).sqrt() as f32;
                (0..hidden_dim).map(|_| rand_random() * scale).collect()
            })
            .collect();

        // B: (hidden_dim × rank) — zero init (standard LoRA)
        let b_weights: Vec<Vec<f32>> = (0..hidden_dim).map(|_| vec![0.0f32; rank]).collect();

        let adapter = LoRAAdapter {
            rank,
            a_weights,
            b_weights,
            target_layers: (0..config.num_layers).collect(),
            alpha: train_config.lora_alpha as f32,
        };

        // Initialize AdamW moments
        let m_a = vec![vec![0.0f32; hidden_dim]; rank];
        let v_a = vec![vec![0.0f32; hidden_dim]; rank];
        let m_b = vec![vec![0.0f32; rank]; hidden_dim];
        let v_b = vec![vec![0.0f32; rank]; hidden_dim];

        Self {
            adapter,
            experience_buffer: Vec::new(),
            fedavg: FedAvgState::default(),
            config,
            train_config,
            node_id,
            step: 0,
            running_loss: f64::MAX,
            best_loss: f64::MAX,
            m_a,
            v_a,
            m_b,
            v_b,
        }
    }

    /// Add a training sample to the experience buffer.
    pub fn add_experience(&mut self, sample: TrainingSample) {
        if sample.reward < self.train_config.min_reward {
            debug!("Skipping sample with low reward: {:.3}", sample.reward);
            return;
        }

        if self.experience_buffer.len() >= self.train_config.max_buffer_size {
            // Remove lowest-reward sample (prioritized replay)
            if let Some(min_idx) = self
                .experience_buffer
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.reward
                        .partial_cmp(&b.1.reward)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if sample.reward > self.experience_buffer[min_idx].reward {
                    self.experience_buffer.remove(min_idx);
                } else {
                    return;
                }
            }
        }
        self.experience_buffer.push(sample);
    }

    /// Run a local training step on collected experience.
    ///
    /// Implements real LoRA training:
    /// 1. Sample a mini-batch from the experience buffer
    /// 2. Forward pass: compute LoRA output ΔW = α/rank * B @ (A @ x)
    /// 3. Compute MSE loss between predicted and target latent vectors
    /// 4. Compute analytical gradients for A and B
    /// 5. Apply AdamW optimizer updates
    /// 6. Return gradient deltas for federated sharing
    pub fn train_step(&mut self) -> Result<Vec<HyphaeMessage>> {
        if self.experience_buffer.is_empty() {
            warn!("No experience to train on");
            return Ok(Vec::new());
        }

        let batch_size = self
            .train_config
            .batch_size
            .min(self.experience_buffer.len());

        // 1. Sample mini-batch (prioritized by reward)
        let batch = self.sample_batch(batch_size);

        // 2. Forward pass + loss computation
        // For each sample: LoRA output = α/rank * B @ (A @ x)
        // Loss = MSE(predicted, target) weighted by reward
        let rank = self.adapter.rank;
        let alpha_over_rank = self.adapter.alpha / rank as f32;
        let input_dim = self.config.hidden_dim;
        let output_dim = self.config.hidden_dim;

        // Accumulate gradients for A and B
        let mut grad_a: Vec<Vec<f32>> = vec![vec![0.0f32; input_dim]; rank];
        let mut grad_b: Vec<Vec<f32>> = vec![vec![0.0f32; rank]; output_dim];
        let mut total_loss = 0.0f64;

        for sample in &batch {
            let input = &sample.input_latent.data;
            let target = &sample.target_latent.data;

            // Compute A @ x: (rank, hidden_dim) @ (hidden_dim,) → (rank,)
            let mut ax = vec![0.0f32; rank];
            for (r, a_row) in self.adapter.a_weights.iter().enumerate() {
                for (j, &a_val) in a_row.iter().enumerate() {
                    if j < input.len() {
                        ax[r] += a_val * input[j];
                    }
                }
            }

            // Compute B @ (A @ x): (output_dim, rank) @ (rank,) → (output_dim,)
            let mut predicted = vec![0.0f32; output_dim];
            for (i, b_row) in self.adapter.b_weights.iter().enumerate() {
                for (r, &b_val) in b_row.iter().enumerate() {
                    predicted[i] += b_val * ax[r];
                }
            }

            // Scale by α/rank
            for p in predicted.iter_mut() {
                *p *= alpha_over_rank;
            }

            // Compute residual and loss
            let mut loss = 0.0f64;
            let mut residual = vec![0.0f32; output_dim];
            for i in 0..output_dim.min(target.len()) {
                residual[i] = predicted[i] - target[i];
                loss += (residual[i] as f64) * (residual[i] as f64);
            }
            let weighted_loss = loss * sample.reward as f64;
            total_loss += weighted_loss;

            // Compute gradients
            // dL/dB[i][r] = 2 * residual[i] * ax[r] * α/rank * reward
            // dL/dA[r][j] = 2 * (Σ_i residual[i] * B[i][r]) * x[j] * α/rank * reward
            let scale = 2.0 * alpha_over_rank * sample.reward;

            // Gradient for A
            #[allow(clippy::needless_range_loop)]
            for r in 0..rank {
                // Compute Σ_i residual[i] * B[i][r]
                let mut sum_residual_b = 0.0f32;
                #[allow(clippy::needless_range_loop)]
                for i in 0..output_dim.min(self.adapter.b_weights.len()) {
                    sum_residual_b += residual[i] * self.adapter.b_weights[i][r];
                }
                for j in 0..input_dim.min(input.len()) {
                    grad_a[r][j] += scale * sum_residual_b * input[j];
                }
            }

            // Gradient for B
            for i in 0..output_dim.min(self.adapter.b_weights.len()) {
                for r in 0..rank {
                    grad_b[i][r] += scale * residual[i] * ax[r];
                }
            }
        }

        // Average gradients over batch
        let n = batch.len() as f32;
        for row in grad_a.iter_mut() {
            for v in row.iter_mut() {
                *v /= n;
            }
        }
        for row in grad_b.iter_mut() {
            for v in row.iter_mut() {
                *v /= n;
            }
        }

        let avg_loss = total_loss / batch.len() as f64;
        self.running_loss = if self.running_loss == f64::MAX {
            avg_loss
        } else {
            0.9 * self.running_loss + 0.1 * avg_loss
        };

        if avg_loss < self.best_loss {
            self.best_loss = avg_loss;
        }

        // 3. Apply AdamW optimizer
        self.adamw_update(&grad_a, &grad_b);

        // 4. Generate gradient deltas for federated sharing
        // Flatten A weights changes as delta
        let mut deltas = Vec::new();
        for &layer_idx in &self.adapter.target_layers {
            // Flatten the A gradient as the delta to share
            let mut flat_delta: Vec<f32> = Vec::with_capacity(rank * input_dim);
            for row in &grad_a {
                flat_delta.extend_from_slice(row);
            }
            // Add DP noise if configured
            if self.train_config.dp_noise_scale > 0.0 {
                for v in flat_delta.iter_mut() {
                    *v += rand::random::<f32>() * self.train_config.dp_noise_scale as f32;
                }
            }
            deltas.push(HyphaeMessage::GradientDelta {
                layer_idx,
                delta: flat_delta,
                version: self.step,
                node_id: self.node_id,
            });
        }

        self.step += 1;

        // Decay learning rate
        self.train_config.learning_rate *= 0.9999;

        debug!(
            "Training step {}: loss={:.6}, running={:.6}, {} deltas",
            self.step,
            avg_loss,
            self.running_loss,
            deltas.len()
        );

        Ok(deltas)
    }

    /// AdamW optimizer update.
    fn adamw_update(&mut self, grad_a: &[Vec<f32>], grad_b: &[Vec<f32>]) {
        let lr = self.train_config.learning_rate as f32;
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let eps: f32 = 1e-8;
        let weight_decay = self.train_config.weight_decay as f32;
        let t = self.step as f32 + 1.0;

        // Update A weights
        for r in 0..self.adapter.a_weights.len() {
            for j in 0..self.adapter.a_weights[r].len() {
                if r >= grad_a.len() || j >= grad_a[r].len() {
                    continue;
                }
                let g = grad_a[r][j];

                // First moment
                self.m_a[r][j] = beta1 * self.m_a[r][j] + (1.0 - beta1) * g;
                // Second moment
                self.v_a[r][j] = beta2 * self.v_a[r][j] + (1.0 - beta2) * g * g;
                // Bias correction
                let m_hat = self.m_a[r][j] / (1.0 - beta1.powf(t));
                let v_hat = self.v_a[r][j] / (1.0 - beta2.powf(t));
                // AdamW update (weight decay is NOT scaled by lr)
                self.adapter.a_weights[r][j] -= lr
                    * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.adapter.a_weights[r][j]);
            }
        }

        // Update B weights
        for i in 0..self.adapter.b_weights.len() {
            for r in 0..self.adapter.b_weights[i].len() {
                if i >= grad_b.len() || r >= grad_b[i].len() {
                    continue;
                }
                let g = grad_b[i][r];

                self.m_b[i][r] = beta1 * self.m_b[i][r] + (1.0 - beta1) * g;
                self.v_b[i][r] = beta2 * self.v_b[i][r] + (1.0 - beta2) * g * g;
                let m_hat = self.m_b[i][r] / (1.0 - beta1.powf(t));
                let v_hat = self.v_b[i][r] / (1.0 - beta2.powf(t));
                self.adapter.b_weights[i][r] -= lr
                    * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.adapter.b_weights[i][r]);
            }
        }
    }

    /// Sample a mini-batch from the experience buffer.
    /// Uses prioritized experience replay (higher reward = higher probability).
    fn sample_batch(&self, batch_size: usize) -> Vec<&TrainingSample> {
        if self.experience_buffer.len() <= batch_size {
            return self.experience_buffer.iter().collect();
        }

        let total_reward: f32 = self
            .experience_buffer
            .iter()
            .map(|s| s.reward.max(0.01))
            .sum();
        let mut selected = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        let mut attempts = 0;
        while selected.len() < batch_size && attempts < batch_size * 3 {
            let idx = (rand::random::<f32>() * self.experience_buffer.len() as f32) as usize;
            let sample = &self.experience_buffer[idx.min(self.experience_buffer.len() - 1)];
            let prob = sample.reward.max(0.01) / total_reward;
            if rand::random::<f32>() < prob * self.experience_buffer.len() as f32 {
                selected.push(sample);
            }
            attempts += 1;
        }

        while selected.len() < batch_size {
            let idx = rand::Rng::gen_range(&mut rng, 0..self.experience_buffer.len());
            selected.push(&self.experience_buffer[idx]);
        }

        selected
    }

    /// Receive a federated gradient delta from a peer.
    pub fn receive_federated_delta(&mut self, msg: &HyphaeMessage) {
        self.fedavg.add_gradient_message(msg);
    }

    /// Try to apply federated averaging if enough peers have contributed.
    pub fn try_federated_average(&mut self) -> Result<bool> {
        if let Some(averaged) = self.fedavg.federated_average() {
            self.apply_deltas(&averaged)?;
            info!(
                "Applied federated average from {} deltas, version {}",
                averaged.len(),
                self.fedavg.global_version,
            );
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Apply gradient deltas to the local LoRA adapter.
    /// Deltas are flattened A-weight gradients, reshaped back to (rank × hidden_dim).
    pub fn apply_deltas(&mut self, deltas: &[(usize, Vec<f32>)]) -> Result<()> {
        let rank = self.adapter.rank;
        let hidden_dim = self.config.hidden_dim;

        for (layer_idx, delta) in deltas {
            if !self.adapter.target_layers.contains(layer_idx) {
                continue;
            }

            // Reshape flat delta back to (rank × hidden_dim) and apply to A weights
            let rows = delta.len() / hidden_dim;
            for r in 0..rows.min(rank) {
                for j in 0..hidden_dim {
                    let flat_idx = r * hidden_dim + j;
                    if flat_idx < delta.len()
                        && r < self.adapter.a_weights.len()
                        && j < self.adapter.a_weights[r].len()
                    {
                        self.adapter.a_weights[r][j] +=
                            delta[flat_idx] * self.adapter.alpha / rank as f32;
                    }
                }
            }
        }
        Ok(())
    }

    /// Generate self-play training samples.
    pub fn self_play(
        &mut self,
        prompt_latent: &LatentVector,
        output_latent: &LatentVector,
        reward: f32,
    ) {
        let sample = TrainingSample {
            input_latent: prompt_latent.clone(),
            target_latent: output_latent.clone(),
            reward,
            source: SampleSource::SelfPlay,
            timestamp: chrono::Utc::now(),
        };
        self.add_experience(sample);
    }

    /// Get the current LoRA adapter.
    pub fn adapter(&self) -> &LoRAAdapter {
        &self.adapter
    }

    /// Get the training step count.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Get the experience buffer size.
    pub fn experience_size(&self) -> usize {
        self.experience_buffer.len()
    }

    /// Get the running loss.
    pub fn running_loss(&self) -> f64 {
        self.running_loss
    }

    /// Get the best loss.
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }

    /// Get the training config.
    pub fn train_config(&self) -> &TrainingConfig {
        &self.train_config
    }
}

/// Simple random number for initialization.
fn rand_random() -> f32 {
    rand::random::<f32>() * 2.0 - 1.0
}

// ─── Gradient Bridge ─────────────────────────────────────────────────────────

/// Wrapper that bridges the Nucleus to inference engines for automatic gradient collection.
///
/// This type wraps a `Nucleus` and provides methods to:
/// - Record inference traces automatically
/// - Convert traces to training samples
/// - Expose training metrics to the API layer
pub struct NucleusWithBridge {
    /// The underlying nucleus for LoRA training
    nucleus: Nucleus,
    /// Collected inference traces
    traces: Vec<InferenceTrace>,
}

/// Records a single inference pass for later gradient computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTrace {
    /// Request ID this trace belongs to
    pub request_id: Uuid,
    /// Input latent vector (before processing)
    pub input_latent: LatentVector,
    /// Output latent vector (after processing)
    pub output_latent: LatentVector,
    /// Which layer this trace was collected at
    pub layer_idx: usize,
    /// Computed reward for this inference pass
    pub reward: f32,
    /// Timestamp of the trace
    pub timestamp: DateTime<Utc>,
}

impl NucleusWithBridge {
    /// Create a new nucleus with gradient bridge.
    pub fn new(config: ModelConfig, node_id: NodeId) -> Self {
        Self {
            nucleus: Nucleus::new(config.clone(), node_id),
            traces: Vec::new(),
        }
    }

    /// Record an inference trace for later training.
    pub fn record_trace(
        &mut self,
        request_id: Uuid,
        input_latent: LatentVector,
        output_latent: LatentVector,
        layer_idx: usize,
        reward: f32,
    ) {
        let trace = InferenceTrace {
            request_id,
            input_latent,
            output_latent,
            layer_idx,
            reward,
            timestamp: Utc::now(),
        };
        self.traces.push(trace);
    }

    /// Get a reference to the gradient bridge (for API integration).
    pub fn gradient_bridge(&self) -> GradientBridge {
        GradientBridge {
            traces: self.traces.clone(),
        }
    }

    /// Create training samples from collected traces and add them to the nucleus.
    pub fn create_training_samples_from_traces(&mut self) -> usize {
        let count = self.traces.len();
        for trace in &self.traces {
            let sample = TrainingSample {
                input_latent: trace.input_latent.clone(),
                target_latent: trace.output_latent.clone(),
                reward: trace.reward,
                source: SampleSource::SelfPlay,
                timestamp: trace.timestamp,
            };
            self.nucleus.add_experience(sample);
        }
        // Clear traces after converting to samples
        self.traces.clear();
        count
    }

    /// Run a training step on collected experience.
    pub fn train_step(&mut self) -> Result<Vec<HyphaeMessage>> {
        self.nucleus.train_step()
    }

    /// Get the training step count.
    pub fn step(&self) -> u64 {
        self.nucleus.step()
    }

    /// Get the experience buffer size.
    pub fn experience_size(&self) -> usize {
        self.nucleus.experience_size()
    }

    /// Get the running loss.
    pub fn running_loss(&self) -> f64 {
        self.nucleus.running_loss()
    }

    /// Get the best loss.
    pub fn best_loss(&self) -> f64 {
        self.nucleus.best_loss()
    }

    /// Add experience directly.
    pub fn add_experience(&mut self, sample: TrainingSample) {
        self.nucleus.add_experience(sample);
    }

    /// Get the current LoRA adapter.
    pub fn adapter(&self) -> &LoRAAdapter {
        self.nucleus.adapter()
    }

    /// Process a batch of reward signals and convert them into training experiences.
    ///
    /// For each signal, finds the matching inference trace (by request ID) and
    /// creates a `TrainingSample` with the signal's computed reward. Signals whose
    /// request ID does not match any recorded trace are silently skipped.
    ///
    /// Returns the number of training samples that were successfully added.
    pub fn process_reward_signals(&mut self, signals: Vec<RewardSignal>) -> usize {
        let mut aggregator = RewardAggregator::with_defaults();
        let mut added = 0usize;

        // First pass: file all signals into the aggregator
        for signal in &signals {
            aggregator.add_signal(signal.clone());
        }

        // Collect unique request IDs from the signals
        let request_ids: Vec<Uuid> = signals
            .iter()
            .map(|s| s.request_id())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // For each request ID, find matching traces and create training samples
        for request_id in &request_ids {
            // Find the most recent trace for this request
            let trace = self
                .traces
                .iter()
                .rev()
                .find(|t| t.request_id == *request_id);

            if let Some(trace) = trace {
                let reward = aggregator
                    .compute_reward(request_id)
                    .unwrap_or(trace.reward);

                let sample = TrainingSample {
                    input_latent: trace.input_latent.clone(),
                    target_latent: trace.output_latent.clone(),
                    reward,
                    source: SampleSource::UserFeedback,
                    timestamp: Utc::now(),
                };

                self.nucleus.add_experience(sample);
                added += 1;
            } else {
                debug!(
                    "process_reward_signals: no trace found for request {}",
                    request_id
                );
            }
        }

        info!(
            "Processed {} reward signals → {} training samples",
            signals.len(),
            added
        );

        added
    }
}

/// Gradient bridge — exposes collected inference traces for training sample creation.
pub struct GradientBridge {
    /// Collected inference traces
    pub traces: Vec<InferenceTrace>,
}

impl GradientBridge {
    /// Create training samples from collected traces.
    pub fn create_training_samples(&self) -> Vec<TrainingSample> {
        self.traces
            .iter()
            .map(|trace| TrainingSample {
                input_latent: trace.input_latent.clone(),
                target_latent: trace.output_latent.clone(),
                reward: trace.reward,
                source: SampleSource::SelfPlay,
                timestamp: trace.timestamp,
            })
            .collect()
    }

    /// Get the number of collected traces.
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }
}

// ─── Reward Signal from Usage ────────────────────────────────────────────────

/// A reward signal derived from user interaction or automatic quality metrics.
///
/// Reward signals are the primary mechanism for the nucleus to learn from usage.
/// They are collected from multiple sources and aggregated into a composite
/// reward that drives LoRA adapter updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardSignal {
    /// Direct user rating on a 0–1 scale.
    UserRating {
        /// Rating score in \[0, 1\].
        score: f32,
        /// The inference request this rating applies to.
        request_id: Uuid,
    },
    /// Implicit feedback based on how long the user engaged with the output.
    ImplicitFeedback {
        /// Engagement duration in milliseconds.
        engagement_ms: u64,
        /// The inference request this feedback applies to.
        request_id: Uuid,
    },
    /// The user accepted the generated output (e.g. applied a suggestion).
    CompletionAccepted {
        /// The inference request that was accepted.
        request_id: Uuid,
    },
    /// The user rejected or regenerated the output.
    CompletionRejected {
        /// The inference request that was rejected.
        request_id: Uuid,
    },
    /// Automatic latent-space quality metrics computed post-inference.
    LatentQuality {
        /// Coherence score in \[0, 1\] — how internally consistent the output is.
        coherence: f32,
        /// Diversity score in \[0, 1\] — how novel the output is relative to recent history.
        diversity: f32,
        /// The inference request these metrics apply to.
        request_id: Uuid,
    },
}

impl RewardSignal {
    /// Extract the request ID from any signal variant.
    pub fn request_id(&self) -> Uuid {
        match self {
            RewardSignal::UserRating { request_id, .. } => *request_id,
            RewardSignal::ImplicitFeedback { request_id, .. } => *request_id,
            RewardSignal::CompletionAccepted { request_id } => *request_id,
            RewardSignal::CompletionRejected { request_id } => *request_id,
            RewardSignal::LatentQuality { request_id, .. } => *request_id,
        }
    }

    /// Compute the raw scalar reward for this signal (before weighting).
    pub fn raw_reward(&self) -> f32 {
        match self {
            RewardSignal::UserRating { score, .. } => score.clamp(0.0, 1.0),
            RewardSignal::ImplicitFeedback { engagement_ms, .. } => {
                // Sigmoid-like mapping: 0ms→0, ~5000ms→0.5, 30000ms+→~1.0
                let t = *engagement_ms as f32 / 10_000.0;
                1.0 - (-t).exp()
            }
            RewardSignal::CompletionAccepted { .. } => 1.0,
            RewardSignal::CompletionRejected { .. } => 0.0,
            RewardSignal::LatentQuality {
                coherence,
                diversity,
                ..
            } => {
                // Geometric mean so both must be decent
                (coherence.clamp(0.0, 1.0) * diversity.clamp(0.0, 1.0)).sqrt()
            }
        }
    }
}

// ─── Reward Aggregator ───────────────────────────────────────────────────────

/// Configuration for how reward signals are weighted and aggregated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardConfig {
    /// Weight for `UserRating` signals.
    pub user_rating_weight: f32,
    /// Weight for `ImplicitFeedback` signals.
    pub implicit_feedback_weight: f32,
    /// Weight for `CompletionAccepted` signals.
    pub completion_accepted_weight: f32,
    /// Weight for `CompletionRejected` signals.
    pub completion_rejected_weight: f32,
    /// Weight for `LatentQuality` signals.
    pub latent_quality_weight: f32,
    /// Exponential decay factor for the running reward EMA (0 < decay < 1).
    pub decay_factor: f32,
    /// Maximum number of requests to keep in history.
    pub max_history: usize,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            user_rating_weight: 1.0,
            implicit_feedback_weight: 0.5,
            completion_accepted_weight: 0.8,
            completion_rejected_weight: 0.8,
            latent_quality_weight: 0.3,
            decay_factor: 0.95,
            max_history: 10_000,
        }
    }
}

/// Aggregates reward signals from multiple sources into composite reward scores.
///
/// Maintains per-request signal history and computes a weighted composite reward
/// that can be fed into the training loop. Also tracks an exponential moving
/// average of rewards for trend monitoring.
pub struct RewardAggregator {
    /// Aggregation configuration / weights.
    config: RewardConfig,
    /// Per-request collected signals.
    history: HashMap<Uuid, Vec<RewardSignal>>,
    /// Insertion-order tracking for eviction.
    insertion_order: Vec<Uuid>,
    /// Exponential moving average of composite rewards.
    ema_reward: f64,
    /// Whether the EMA has been initialised.
    ema_initialised: bool,
}

impl RewardAggregator {
    /// Create a new aggregator with the given configuration.
    pub fn new(config: RewardConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            insertion_order: Vec::new(),
            ema_reward: 0.0,
            ema_initialised: false,
        }
    }

    /// Create a new aggregator with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(RewardConfig::default())
    }

    /// Add a reward signal. The signal is filed under its request ID.
    pub fn add_signal(&mut self, signal: RewardSignal) {
        let request_id = signal.request_id();

        // Track insertion order for eviction
        if !self.history.contains_key(&request_id) {
            self.insertion_order.push(request_id);
        }

        self.history.entry(request_id).or_default().push(signal);

        // Evict oldest entries if we exceed max history
        while self.insertion_order.len() > self.config.max_history {
            if let Some(old_id) = self.insertion_order.first().copied() {
                self.insertion_order.remove(0);
                self.history.remove(&old_id);
            }
        }
    }

    /// Compute the composite reward for a specific request.
    ///
    /// Returns `None` if no signals have been recorded for that request.
    /// The composite is a weighted average of individual signal rewards,
    /// normalised by total weight so the result stays in \[0, 1\].
    pub fn compute_reward(&mut self, request_id: &Uuid) -> Option<f32> {
        let signals = self.history.get(request_id)?;
        if signals.is_empty() {
            return None;
        }

        let mut weighted_sum = 0.0f32;
        let mut total_weight = 0.0f32;

        for signal in signals {
            let (raw, weight) = match signal {
                RewardSignal::UserRating { .. } => {
                    (signal.raw_reward(), self.config.user_rating_weight)
                }
                RewardSignal::ImplicitFeedback { .. } => {
                    (signal.raw_reward(), self.config.implicit_feedback_weight)
                }
                RewardSignal::CompletionAccepted { .. } => {
                    (signal.raw_reward(), self.config.completion_accepted_weight)
                }
                RewardSignal::CompletionRejected { .. } => {
                    (signal.raw_reward(), self.config.completion_rejected_weight)
                }
                RewardSignal::LatentQuality { .. } => {
                    (signal.raw_reward(), self.config.latent_quality_weight)
                }
            };
            weighted_sum += raw * weight;
            total_weight += weight;
        }

        let composite = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };

        // Update EMA
        if self.ema_initialised {
            self.ema_reward = self.config.decay_factor as f64 * self.ema_reward
                + (1.0 - self.config.decay_factor as f64) * composite as f64;
        } else {
            self.ema_reward = composite as f64;
            self.ema_initialised = true;
        }

        Some(composite)
    }

    /// Get all signals recorded for a given request.
    pub fn get_reward_for_request(&self, request_id: &Uuid) -> Option<&Vec<RewardSignal>> {
        self.history.get(request_id)
    }

    /// Get the current exponential moving average of rewards.
    pub fn ema_reward(&self) -> f64 {
        self.ema_reward
    }

    /// Get the number of tracked requests.
    pub fn tracked_requests(&self) -> usize {
        self.history.len()
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &RewardConfig {
        &self.config
    }
}

// ─── Usage Tracker ───────────────────────────────────────────────────────────

/// Tracks inference requests and their outcomes, automatically computing
/// implicit rewards and generating training samples.
///
/// The `UsageTracker` sits between the inference engine and the `NucleusWithBridge`,
/// observing request/response pairs and user actions to produce `TrainingSample`s
/// with computed composite rewards.
pub struct UsageTracker {
    /// Per-request metadata needed to compute implicit rewards.
    pending_requests: HashMap<Uuid, UsageRecord>,
    /// The reward aggregator that computes composite scores.
    aggregator: RewardAggregator,
    /// Maximum pending requests before we start evicting stale entries.
    max_pending: usize,
}

/// Internal record for an in-flight inference request.
#[derive(Debug, Clone)]
struct UsageRecord {
    /// Input latent vector captured at request time.
    input_latent: LatentVector,
    /// Output latent vector captured at response time (None until response).
    output_latent: Option<LatentVector>,
    /// When the request was issued.
    request_time: DateTime<Utc>,
    /// When the response was delivered (None until response).
    response_time: Option<DateTime<Utc>>,
    /// Total tokens generated (set at response time).
    tokens_generated: u64,
}

impl UsageTracker {
    /// Create a new usage tracker with the given reward config.
    pub fn new(reward_config: RewardConfig) -> Self {
        Self {
            pending_requests: HashMap::new(),
            aggregator: RewardAggregator::new(reward_config),
            max_pending: 50_000,
        }
    }

    /// Create a new usage tracker with default reward config.
    pub fn with_defaults() -> Self {
        Self::new(RewardConfig::default())
    }

    /// Track a new inference request.
    pub fn track_request(&mut self, request_id: Uuid, input_latent: LatentVector) {
        // Evict stale entries if we're at capacity
        if self.pending_requests.len() >= self.max_pending {
            // Remove oldest entry
            let oldest = self.pending_requests.keys().next().copied();
            if let Some(id) = oldest {
                self.pending_requests.remove(&id);
            }
        }

        self.pending_requests.insert(
            request_id,
            UsageRecord {
                input_latent,
                output_latent: None,
                request_time: Utc::now(),
                response_time: None,
                tokens_generated: 0,
            },
        );
    }

    /// Track the response for a previously tracked request.
    pub fn track_response(
        &mut self,
        request_id: Uuid,
        output_latent: LatentVector,
        tokens_generated: u64,
    ) {
        if let Some(record) = self.pending_requests.get_mut(&request_id) {
            record.output_latent = Some(output_latent);
            record.response_time = Some(Utc::now());
            record.tokens_generated = tokens_generated;

            // Automatically generate an implicit feedback signal from response latency.
            let latency_ms = record
                .response_time
                .unwrap()
                .signed_duration_since(record.request_time)
                .num_milliseconds()
                .unsigned_abs();

            // Lower latency → higher implicit reward (inverse sigmoid)
            let latency_reward = 1.0 / (1.0 + (latency_ms as f32 / 2000.0));

            // Token acceptance rate as a proxy signal: more tokens → more engagement
            let token_signal = (tokens_generated as f32 / 500.0).min(1.0);

            // Average the two implicit signals
            let engagement_ms = ((latency_reward + token_signal) / 2.0 * 10_000.0) as u64;

            self.aggregator.add_signal(RewardSignal::ImplicitFeedback {
                engagement_ms,
                request_id,
            });
        } else {
            debug!("track_response called for unknown request: {}", request_id);
        }
    }

    /// Track a user action (accept, reject, rating) for a request.
    pub fn track_user_action(&mut self, signal: RewardSignal) {
        self.aggregator.add_signal(signal);
    }

    /// Generate `TrainingSample`s for all completed requests that have at least one signal.
    ///
    /// Consumes the pending requests that have both input and output latents,
    /// computes composite rewards, and returns training samples ready for the nucleus.
    pub fn generate_training_samples(&mut self) -> Vec<TrainingSample> {
        let mut samples = Vec::new();

        // Collect completed request IDs
        let completed_ids: Vec<Uuid> = self
            .pending_requests
            .iter()
            .filter(|(_, r)| r.output_latent.is_some())
            .map(|(id, _)| *id)
            .collect();

        for request_id in completed_ids {
            if let Some(record) = self.pending_requests.remove(&request_id) {
                let reward = self.aggregator.compute_reward(&request_id).unwrap_or(0.5); // default neutral reward

                if let Some(output_latent) = record.output_latent {
                    samples.push(TrainingSample {
                        input_latent: record.input_latent,
                        target_latent: output_latent,
                        reward,
                        source: SampleSource::UserFeedback,
                        timestamp: record.response_time.unwrap_or_else(Utc::now),
                    });
                }
            }
        }

        info!(
            "Generated {} training samples from usage tracking (EMA reward: {:.4})",
            samples.len(),
            self.aggregator.ema_reward()
        );

        samples
    }

    /// Get a reference to the underlying reward aggregator.
    pub fn aggregator(&self) -> &RewardAggregator {
        &self.aggregator
    }

    /// Get a mutable reference to the underlying reward aggregator.
    pub fn aggregator_mut(&mut self) -> &mut RewardAggregator {
        &mut self.aggregator
    }

    /// Get the number of pending (in-flight) requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nucleus_creation() {
        let config = ModelConfig::minimax_m25();
        let nucleus = Nucleus::new(config.clone(), NodeId::new());
        assert_eq!(nucleus.adapter().rank, LORA_DEFAULT_RANK);
        assert_eq!(nucleus.experience_size(), 0);
    }

    #[test]
    fn test_federated_averaging() {
        let mut state = FedAvgState::default();

        for _ in 0..3 {
            state.add_peer_delta(0, vec![1.0, 2.0], 1, NodeId::new());
        }

        assert!(state.round_complete);
        let result = state.federated_average();
        assert!(result.is_some());
        assert_eq!(state.global_version, 1);
    }

    #[test]
    fn test_self_play() {
        let config = ModelConfig::minimax_m25();
        let mut nucleus = Nucleus::new(config.clone(), NodeId::new());

        let prompt = LatentVector::zeros(6144, 0, Uuid::new_v4());
        let output = LatentVector::zeros(6144, 1, Uuid::new_v4());
        nucleus.self_play(&prompt, &output, 0.8);

        assert_eq!(nucleus.experience_size(), 1);
    }

    #[test]
    fn test_train_step() {
        let config = ModelConfig::minimax_m25();
        let mut nucleus = Nucleus::new(config.clone(), NodeId::new());

        for i in 0..10 {
            let prompt = LatentVector::zeros(6144, 0, Uuid::new_v4());
            let mut target_data = vec![0.0f32; 6144];
            target_data[i] = 1.0;
            let target = LatentVector::from_vec(target_data, 1, Uuid::new_v4());
            nucleus.add_experience(TrainingSample {
                input_latent: prompt,
                target_latent: target,
                reward: 0.8,
                source: SampleSource::SelfPlay,
                timestamp: chrono::Utc::now(),
            });
        }

        assert_eq!(nucleus.experience_size(), 10);
        let deltas = nucleus.train_step().unwrap();
        assert!(!deltas.is_empty());
        assert_eq!(nucleus.step(), 1);
    }

    #[test]
    fn test_min_reward_filter() {
        let config = ModelConfig::minimax_m25();
        let mut train_config = TrainingConfig::default();
        train_config.min_reward = 0.5;
        let mut nucleus = Nucleus::with_train_config(config.clone(), NodeId::new(), train_config);

        let prompt = LatentVector::zeros(6144, 0, Uuid::new_v4());
        let target = LatentVector::zeros(6144, 1, Uuid::new_v4());
        nucleus.add_experience(TrainingSample {
            input_latent: prompt.clone(),
            target_latent: target.clone(),
            reward: 0.1,
            source: SampleSource::UserFeedback,
            timestamp: chrono::Utc::now(),
        });
        assert_eq!(nucleus.experience_size(), 0);

        nucleus.add_experience(TrainingSample {
            input_latent: prompt,
            target_latent: target,
            reward: 0.9,
            source: SampleSource::UserFeedback,
            timestamp: chrono::Utc::now(),
        });
        assert_eq!(nucleus.experience_size(), 1);
    }

    #[test]
    fn test_lora_forward_pass() {
        let config = ModelConfig::minimax_m25();
        let nucleus = Nucleus::new(config.clone(), NodeId::new());

        let adapter = nucleus.adapter();
        assert_eq!(adapter.a_weights.len(), adapter.rank);
        assert_eq!(adapter.a_weights[0].len(), config.hidden_dim);
        assert_eq!(adapter.b_weights.len(), config.hidden_dim);
        assert_eq!(adapter.b_weights[0].len(), adapter.rank);
    }

    #[test]
    fn test_apply_deltas() {
        let config = ModelConfig::minimax_m25();
        let mut nucleus = Nucleus::new(config.clone(), NodeId::new());

        let a_before = nucleus.adapter().a_weights[0][0];

        let mut delta = vec![0.0f32; config.hidden_dim * nucleus.adapter().rank];
        delta[0] = 0.1;
        nucleus.apply_deltas(&[(0, delta)]).unwrap();

        assert_ne!(nucleus.adapter().a_weights[0][0], a_before);
    }
}
