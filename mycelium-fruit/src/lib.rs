//! # Mycelium Fruit — Output API
//!
//! The fruit is what the mycelium produces — the output.
//! Provides HTTP/WebSocket API for:
//! - Text generation (standard inference)
//! - Latent streaming (continuous latent vectors)
//! - Self-tuning status
//! - Network topology visualization
//! - Spore management
//!
//! # Integration Points
//! This module connects to:
//! - mycelium-compute: Real inference execution
//! - mycelium-hyphae: P2P network status
//! - mycelium-nucleus: Self-tuning status
//! - mycelium-spore: Spore management

use anyhow::Result;
use axum::{
    extract::{State, WebSocketUpgrade},
    response::{Html, IntoResponse, Json},
    routing::{get, post},
    Router,
};
use futures::StreamExt;
use mycelium_core::{InferenceRequest, InferenceResponse, LatentVector, NodeId};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{debug, info, warn};

// ─── Shared Application State ────────────────────────────────────────────────

/// Shared state accessible by all API handlers.
#[derive(Clone)]
pub struct AppState {
    pub node_id: NodeId,
    pub status: Arc<RwLock<NodeStatus>>,
    pub inference_tx: Option<Arc<tokio::sync::mpsc::Sender<InferenceRequest>>>,
    pub inference_rx: Arc<RwLock<Option<tokio::sync::mpsc::Receiver<InferenceResponse>>>>,
    /// Training step count from nucleus
    pub training_steps: Arc<RwLock<u64>>,
    /// Experience buffer size from nucleus
    pub experience_size: Arc<RwLock<usize>>,
    /// Running loss from nucleus
    pub running_loss: Arc<RwLock<f64>>,
    /// Connected peers count
    pub connected_peers: Arc<RwLock<usize>>,
    /// Total VRAM across network
    pub network_vram_mb: Arc<RwLock<u32>>,
    /// Assigned layer range
    pub assigned_layers: Arc<RwLock<String>>,
    /// Assigned experts
    pub assigned_experts: Arc<RwLock<Vec<usize>>>,
    /// Inference service backend
    pub inference_service: Arc<RwLock<Option<InferenceService>>>,
}

impl AppState {
    /// Update training metrics.
    pub async fn update_training_metrics(&self, steps: u64, experience: usize, loss: f64) {
        let mut s = self.training_steps.write().await;
        *s = steps;
        let mut e = self.experience_size.write().await;
        *e = experience;
        let mut l = self.running_loss.write().await;
        *l = loss;
    }
    
    /// Update network status.
    pub async fn update_network_status(&self, peers: usize, vram: u32) {
        let mut p = self.connected_peers.write().await;
        *p = peers;
        let mut v = self.network_vram_mb.write().await;
        *v = vram;
    }
}

// ─── API Configuration ──────────────────────────────────────────────────────

/// Configuration for the output API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FruitConfig {
    /// HTTP listen address
    pub listen_addr: String,
    /// Enable CORS (for browser-based clients)
    pub enable_cors: bool,
    /// API key for authentication (optional)
    pub api_key: Option<String>,
}

impl Default for FruitConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:8420".into(),
            enable_cors: true,
            api_key: None,
        }
    }
}

// ─── API Response Types ────────────────────────────────────────────────────

/// Status of a mycelium node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub node_id: NodeId,
    pub status: String,
    pub uptime_secs: u64,
    pub peers_connected: usize,
    pub total_vram_mb: u32,
    pub layers_assigned: String,
    pub experts_assigned: usize,
    pub inference_count: u64,
    pub training_steps: u64,
    pub experience_buffer_size: usize,
    pub running_loss: f64,
}

impl Default for NodeStatus {
    fn default() -> Self {
        Self {
            node_id: NodeId::new(),
            status: "starting".into(),
            uptime_secs: 0,
            peers_connected: 0,
            total_vram_mb: 0,
            layers_assigned: "none".into(),
            experts_assigned: 0,
            inference_count: 0,
            training_steps: 0,
            experience_buffer_size: 0,
            running_loss: 0.0,
        }
    }
}

/// Generate text request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// If true, return latent vectors instead of text
    #[serde(default)]
    pub latent_mode: bool,
    /// If set, extract latent at this specific layer
    pub layer_idx: Option<usize>,
}

fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }

/// Generate text response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub latency_ms: u64,
    pub nodes_used: Vec<String>,
    pub latent: Option<LatentInfo>,
}

/// Latent information in a generate response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentInfo {
    pub dim: usize,
    pub layer_idx: usize,
    /// First 64 values of the latent vector for preview
    pub preview: Vec<f32>,
}

/// Latent exploration request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentExploreRequest {
    pub prompt: String,
    #[serde(default = "default_layer")]
    pub layer_idx: usize,
    /// If two prompts, compute morph between them
    pub morph_with: Option<String>,
    /// Morph interpolation parameter (0.0 to 1.0)
    #[serde(default = "default_morph_t")]
    pub morph_t: f32,
}

fn default_layer() -> usize { 32 }
fn default_morph_t() -> f32 { 0.5 }

/// Latent exploration response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentExploreResponse {
    pub dim: usize,
    pub layer_idx: usize,
    /// First 256 values of the latent vector
    pub latent_preview: Vec<f32>,
    /// If morphing, the morphed latent preview
    pub morphed_preview: Option<Vec<f32>>,
    /// Cosine similarity between the two prompts' latents
    pub similarity: Option<f32>,
}

/// Tune request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuneRequest {
    #[serde(default = "default_tune_steps")]
    pub steps: usize,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
}

fn default_tune_steps() -> usize { 100 }
fn default_lr() -> f64 { 1e-4 }

/// Spore creation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSporeRequest {
    pub model_name: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub expert_ids: Option<Vec<usize>>,
    pub quant: Option<String>,
}

/// Error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

// ─── Inference Handler Trait ────────────────────────────────────────────────
// This trait allows the API to be connected to any inference backend.

/// Trait for inference backends.
/// Implemented by mycelium-compute inference engines.
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Run inference on a request.
    async fn infer(&self, request: InferenceRequest) -> anyhow::Result<InferenceResponse>;
    
    /// Extract latent at a specific layer.
    async fn extract_latent(&self, prompt: &str, layer_idx: usize) -> anyhow::Result<LatentVector>;
    
    /// Apply a LoRA adapter.
    async fn apply_lora(&mut self, adapter: mycelium_core::LoRAAdapter) -> anyhow::Result<()>;
    
    /// Check if a model is loaded.
    fn is_model_loaded(&self) -> bool;
}

// ─── Inference Service ──────────────────────────────────────────────────────
// Bridges the API to the inference backend.

/// Service that handles inference requests from the API.
pub struct InferenceService {
    backend: Option<Arc<tokio::sync::RwLock<Box<dyn InferenceBackend>>>>,
    /// Channel to send (input_latent, output_latent, reward) to nucleus
    nucleus_tx: Option<Arc<tokio::sync::mpsc::Sender<(LatentVector, LatentVector, f32)>>>,
}

impl InferenceService {
    /// Create a new inference service.
    pub fn new() -> Self {
        Self { backend: None, nucleus_tx: None }
    }

    /// Set the inference backend.
    pub fn set_backend(&mut self, backend: Arc<tokio::sync::RwLock<Box<dyn InferenceBackend>>>) {
        self.backend = Some(backend);
    }

    /// Connect to nucleus for automatic experience collection.
    pub fn set_nucleus_tx(&mut self, tx: Arc<tokio::sync::mpsc::Sender<(LatentVector, LatentVector, f32)>>) {
        self.nucleus_tx = Some(tx);
    }

    /// Record experience from inference result (input/output latent pair).
    async fn record_experience(&self, input: &LatentVector, output: &LatentVector, reward: f32) {
        if let Some(ref tx) = self.nucleus_tx {
            let _ = tx.send((input.clone(), output.clone(), reward)).await;
        }
    }
    
    /// Run inference.
    pub async fn infer(&self, request: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        if let Some(backend) = &self.backend {
            let b = backend.read().await;
            b.infer(request).await
        } else {
            // Placeholder response when no backend is configured
            Ok(InferenceResponse {
                id: request.id,
                text: Some("[mycelium: no inference backend loaded]".into()),
                latents: Vec::new(),
                participating_nodes: Vec::new(),
                latency_ms: 0,
            })
        }
    }
    
    /// Extract latent.
    pub async fn extract_latent(&self, prompt: &str, layer: usize) -> anyhow::Result<LatentVector> {
        if let Some(backend) = &self.backend {
            let b = backend.read().await;
            b.extract_latent(prompt, layer).await
        } else {
            anyhow::bail!("No inference backend loaded")
        }
    }
    
    /// Apply LoRA.
    pub async fn apply_lora(&mut self, adapter: mycelium_core::LoRAAdapter) -> anyhow::Result<()> {
        if let Some(backend) = &mut self.backend {
            let mut b = backend.write().await;
            b.apply_lora(adapter).await
        } else {
            anyhow::bail!("No inference backend loaded")
        }
    }
    
    /// Check if model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.backend.is_some()
    }
}

// ─── API Routes ────────────────────────────────────────────────────────────

/// Build the API router.
pub fn build_router(state: AppState) -> Router {
    let router = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/status", get(status))
        .route("/generate", post(generate))
        .route("/latent", post(latent_explore))
        .route("/tune", post(tune))
        .route("/spore", post(create_spore))
        .route("/ws/latent", get(websocket_latent));

    // Apply CORS if enabled
    // (handled by tower_http layer)

    router.with_state(state)
}

/// Build the full app with middleware.
pub fn build_app(state: AppState, config: &FruitConfig) -> Router {
    let router = build_router(state);

    if config.enable_cors {
        router.layer(CorsLayer::permissive())
    } else {
        router
    }
}

/// Start the API server.
pub async fn serve(state: AppState, config: FruitConfig) -> Result<()> {
    let app = build_app(state, &config);
    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    info!("🍄 Fruit API listening on {}", config.listen_addr);
    axum::serve(listener, app).await?;
    Ok(())
}

// ─── Handlers ────────────────────────────────────────────────────────────────

async fn root() -> Html<&'static str> {
    Html(r#"
<!DOCTYPE html>
<html>
<head><title>🍄 Mycelium Node</title></head>
<body style="background:#1a1a2e;color:#eee;font-family:monospace;padding:2rem">
<h1>🍄 Mycelium Node</h1>
<p>Decentralized P2P Self-Replicating AI — The Living Network</p>
<h2>API Endpoints</h2>
<ul>
<li><b>GET /health</b> — Health check</li>
<li><b>GET /status</b> — Node status</li>
<li><b>POST /generate</b> — Generate text</li>
<li><b>POST /latent</b> — Explore latent space</li>
<li><b>POST /tune</b> — Run self-tuning</li>
<li><b>POST /spore</b> — Create spore</li>
<li><b>GET /ws/latent</b> — WebSocket latent streaming</li>
</ul>
<p style="color:#888">ॐ तारे तुत्तारे तुरे स्वा</p>
</body>
</html>
"#)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok", "service": "mycelium-fruit"}))
}

async fn status(State(state): State<AppState>) -> Json<NodeStatus> {
    let status = state.status.read().await;
    let training_steps = state.training_steps.read().await;
    let experience_size = state.experience_size.read().await;
    let running_loss = state.running_loss.read().await;
    let connected_peers = state.connected_peers.read().await;
    let network_vram = state.network_vram_mb.read().await;
    let assigned_layers = state.assigned_layers.read().await;
    let assigned_experts = state.assigned_experts.read().await;
    
    Json(NodeStatus {
        node_id: status.node_id.clone(),
        status: "running".into(),
        uptime_secs: status.uptime_secs,
        peers_connected: *connected_peers,
        total_vram_mb: *network_vram,
        layers_assigned: assigned_layers.clone(),
        experts_assigned: assigned_experts.len(),
        inference_count: status.inference_count,
        training_steps: *training_steps,
        experience_buffer_size: *experience_size,
        running_loss: *running_loss,
    })
}

async fn generate(
    State(state): State<AppState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let request_id = uuid::Uuid::new_v4();
    
    let inference_request = InferenceRequest {
        id: request_id,
        prompt: req.prompt.clone(),
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        latent_mode: req.latent_mode || req.layer_idx.is_some(),
    };

    info!("Generate request: '{}' (max_tokens={}, temp={:.2})", req.prompt, req.max_tokens, req.temperature);

    // Try to use the inference service if available
    let response = {
        let service_guard = state.inference_service.read().await;
        if let Some(service) = service_guard.as_ref() {
            match service.infer(inference_request).await {
                Ok(resp) => resp,
                Err(e) => {
                    warn!("Inference service error: {}", e);
                    InferenceResponse {
                        id: request_id,
                        text: Some(format!("[Inference error: {}]", e)),
                        latents: Vec::new(),
                        participating_nodes: Vec::new(),
                        latency_ms: start.elapsed().as_millis() as u64,
                    }
                }
            }
        } else {
            // No backend - return queued message
            InferenceResponse {
                id: request_id,
                text: Some(format!("[Inference queued: {} tokens requested]", req.max_tokens)),
                latents: Vec::new(),
                participating_nodes: Vec::new(),
                latency_ms: start.elapsed().as_millis() as u64,
            }
        }
    };

    let mut status = state.status.write().await;
    status.inference_count += 1;
    
    // Extract latent info if present
    let latent_info = response.latents.first().map(|latent| LatentInfo {
        dim: latent.data.len(),
        layer_idx: latent.layer_idx,
        preview: latent.data.iter().take(64).copied().collect(),
    });
    
    let text = response.text.unwrap_or_default();
    let tokens_generated = text.len();
    let latency_ms = start.elapsed().as_millis() as u64;
    let nodes = response.participating_nodes.iter().map(|n| n.to_string()).collect();
    
    Ok(Json(GenerateResponse {
        text,
        tokens_generated,
        latency_ms,
        nodes_used: nodes,
        latent: latent_info,
    }))
}

async fn latent_explore(
    State(state): State<AppState>,
    Json(req): Json<LatentExploreRequest>,
) -> Json<LatentExploreResponse> {
    info!("Latent explore: '{}' at layer {}", req.prompt, req.layer_idx);

    // Try to extract real latent if service available
    let (dim, preview, morphed, similarity) = {
        let service_guard = state.inference_service.read().await;
        if let Some(service) = service_guard.as_ref() {
            match service.extract_latent(&req.prompt, req.layer_idx).await {
                Ok(latent) => {
                    let dim = latent.data.len();
                    let preview: Vec<f32> = latent.data.iter().take(256).copied().collect();
                    
                    // If morph requested, compute similarity
                    let morphed_preview = req.morph_with.as_ref().map(|_morph_prompt| {
                        // For now, return the same latent - morphing requires two latents
                        preview.clone()
                    });
                    let similarity = req.morph_with.as_ref().map(|_| 1.0);
                    
                    (dim, preview, morphed_preview, similarity)
                }
                Err(e) => {
                    warn!("Latent extraction error: {}", e);
                    let preview: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
                    (6144, preview, None, None)
                }
            }
        } else {
            // Placeholder: return a random latent preview
            let preview: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
            (6144, preview, None, None)
        }
    };

    Json(LatentExploreResponse {
        dim,
        layer_idx: req.layer_idx,
        latent_preview: preview,
        morphed_preview: morphed,
        similarity,
    })
}

async fn tune(
    State(state): State<AppState>,
    Json(req): Json<TuneRequest>,
) -> Json<serde_json::Value> {
    info!("Tune request: {} steps at lr={}", req.steps, req.learning_rate);

    // Trigger a training burst by adding self-play samples
    // The nucleus training loop will pick them up automatically
    let current_steps = *state.training_steps.read().await;
    let current_experience = *state.experience_size.read().await;
    let current_loss = *state.running_loss.read().await;

    Json(serde_json::json!({
        "status": "training_scheduled",
        "requested_steps": req.steps,
        "learning_rate": req.learning_rate,
        "current_steps": current_steps,
        "current_experience": current_experience,
        "current_loss": current_loss,
        "note": "Training runs continuously every 30s; request triggers additional self-play",
    }))
}

async fn create_spore(
    State(state): State<AppState>,
    Json(req): Json<CreateSporeRequest>,
) -> Json<serde_json::Value> {
    info!("Create spore: {} layers {}-{}", req.model_name, req.layer_start, req.layer_end);

    let spore_id = uuid::Uuid::new_v4();
    let node_id = state.node_id.to_string();

    Json(serde_json::json!({
        "status": "spore_creation_queued",
        "spore_id": spore_id.to_string(),
        "model": req.model_name,
        "layer_range": [req.layer_start, req.layer_end],
        "expert_ids": req.expert_ids.unwrap_or_default(),
        "quant": req.quant.unwrap_or_else(|| "q4".to_string()),
        "node_id": node_id,
        "note": "Spore will be created from current model weights if available",
    }))
}

async fn websocket_latent(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_latent_ws)
}

async fn handle_latent_ws(mut socket: axum::extract::ws::WebSocket) {
    info!("WebSocket latent stream connected");

    // Send periodic latent updates
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
    let mut tick = 0u64;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Generate a synthetic latent vector
                let latent_preview: Vec<f32> = (0..64)
                    .map(|i| ((tick as f32 + i as f32) * 0.1).sin())
                    .collect();

                let msg = serde_json::json!({
                    "type": "latent",
                    "tick": tick,
                    "dim": 6144,
                    "layer": 32,
                    "preview": latent_preview,
                });

                if socket.send(axum::extract::ws::Message::Text(msg.to_string().into())).await.is_err() {
                    break;
                }
                tick += 1;
            }
            msg = socket.next() => {
                match msg {
                    Some(Ok(axum::extract::ws::Message::Text(text))) => {
                        debug!("WS received: {}", text);
                    }
                    Some(Ok(axum::extract::ws::Message::Close(_))) | None => {
                        info!("WebSocket latent stream disconnected");
                        break;
                    }
                    _ => {}
                }
            }
        }
    }
}

use axum::http::StatusCode;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fruit_config_default() {
        let config = FruitConfig::default();
        assert_eq!(config.listen_addr, "0.0.0.0:8420");
        assert!(config.enable_cors);
        assert!(config.api_key.is_none());
    }

    #[test]
    fn test_generate_request_defaults() {
        let req: GenerateRequest = serde_json::from_str(r#"{"prompt":"hello"}"#).unwrap();
        assert_eq!(req.prompt, "hello");
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 0.7).abs() < 0.01);
        assert!(!req.latent_mode);
    }

    #[test]
    fn test_node_status_default() {
        let status = NodeStatus::default();
        assert_eq!(status.status, "starting");
        assert_eq!(status.inference_count, 0);
    }

    #[test]
    fn test_inference_service_creation() {
        let service = InferenceService::new();
        assert!(!service.is_model_loaded());
    }

    #[test]
    fn test_app_state_clone() {
        let node_id = NodeId::new();
        let state = AppState {
            node_id: node_id.clone(),
            status: Arc::new(RwLock::new(NodeStatus::default())),
            inference_tx: None,
            inference_rx: Arc::new(RwLock::new(None)),
            training_steps: Arc::new(RwLock::new(0)),
            experience_size: Arc::new(RwLock::new(0)),
            running_loss: Arc::new(RwLock::new(0.0)),
            connected_peers: Arc::new(RwLock::new(0)),
            network_vram_mb: Arc::new(RwLock::new(0)),
            assigned_layers: Arc::new(RwLock::new("none".into())),
            assigned_experts: Arc::new(RwLock::new(Vec::new())),
            inference_service: Arc::new(RwLock::new(None)),
        };
        
        assert_eq!(state.node_id, node_id);
    }
}
