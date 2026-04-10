//! # Mycelium Node — Main Binary
//!
//! The node is the primary entry point for running a mycelium node.
//! It starts the P2P network, loads models, and serves the Fruit API.
//! All components are wired together:
//! - Hyphae events are dispatched to nucleus (gradients), spore propagator, and tensor router
//! - The Fruit API server connects to real inference, nucleus training, and spore creation
//! - Nucleus runs a training event loop collecting inference traces
//! - Spore propagator monitors capacity and replicates when appropriate

use anyhow::Result;
use clap::{Parser, Subcommand};
use mycelium_core::{ModelConfig, NodeId, HyphaeMessage, SporeGenome, NodeCapabilities};
use mycelium_fruit::{AppState, NodeStatus, FruitConfig, InferenceService, InferenceBackend};
use mycelium_spore::{SporePropagator, PropagationConfig, NodeCapacityState, SporeBuilder};
use tracing::{info, warn, error, debug};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};

// ─── CLI ─────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "mycelium", about = "Decentralized P2P Self-Replicating AI")]
struct Cli {
    /// Path to GGUF model file
    #[arg(long)]
    model_path: Option<std::path::PathBuf>,

    /// Path to tokenizer file (JSON format from HuggingFace)
    #[arg(long)]
    tokenizer_path: Option<std::path::PathBuf>,

    /// Data directory for substrate storage
    #[arg(long, default_value = "~/.mycelium/data")]
    data_dir: String,

    /// P2P listen address
    #[arg(long, default_value = "/ip4/0.0.0.0/0/tcp/4001")]
    listen: String,

    /// Bootstrap P2P peers
    #[arg(long)]
    bootstrap: Vec<String>,

    /// API server port
    #[arg(long, default_value_t = 8080)]
    api_port: u16,

    /// Enable latent streaming mode
    #[arg(long)]
    latent_mode: bool,

    /// Enable spore self-replication mode
    #[arg(long)]
    spore_mode: bool,

    /// Subcommand
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Generate text from a prompt
    Generate {
        /// The prompt text
        prompt: String,
        /// Maximum tokens to generate
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
    },
    /// Explore latent space
    LatentExplore {
        /// The prompt text
        prompt: String,
        /// Layer index to extract from
        #[arg(long, default_value_t = 0)]
        layer: usize,
        /// Morph with another latent
        #[arg(long)]
        morph_with: Option<String>,
    },
    /// Manage spores
    Spore {
        /// Action: create, list, germinate
        action: String,
    },
    /// Show node status
    Status,
}

fn print_banner() {
    eprintln!();
    eprintln!("  ╭──────────────────────────────────────────╮");
    eprintln!("  │  MYCELIUM NODE                           │");
    eprintln!("  │  Decentralized P2P Self-Replicating AI   │");
    eprintln!("  │  v0.1.0 — The Living Network             │");
    eprintln!("  ╰──────────────────────────────────────────╯");
    eprintln!();
    eprintln!("  ॐ तारे तुत्तारे तुरे स्वा");
    eprintln!();
}

// ─── Backend wrapper bridging mycelium-compute to Fruit API ───

/// Wraps the InferenceEngine to implement the InferenceBackend trait for Fruit.
struct ComputeBackend {
    engine: Arc<RwLock<mycelium_compute::InferenceEngine>>,
}

#[async_trait::async_trait]
impl InferenceBackend for ComputeBackend {
    async fn infer(
        &self,
        request: mycelium_core::InferenceRequest,
    ) -> anyhow::Result<mycelium_core::InferenceResponse> {
        let mut eng = self.engine.write().await;
        eng.infer(&request)
    }

    async fn extract_latent(
        &self,
        prompt: &str,
        layer_idx: usize,
    ) -> anyhow::Result<mycelium_core::LatentVector> {
        let mut eng = self.engine.write().await;
        eng.extract_latent(prompt, layer_idx)
    }

    async fn apply_lora(
        &mut self,
        adapter: mycelium_core::LoRAAdapter,
    ) -> anyhow::Result<()> {
        let mut eng = self.engine.write().await;
        eng.apply_lora(&adapter)
    }

    fn is_model_loaded(&self) -> bool {
        true
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI args
    let cli = Cli::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    print_banner();

    let node_id = NodeId::new();
    info!("Node ID: {}", node_id);

    // Handle subcommands
    if let Some(command) = cli.command.clone() {
        handle_command(command, &cli, &node_id).await?;
        return Ok(());
    }

    // ─── Full Node Startup ──────────────────────────────────────────

    info!("Starting mycelium node...");

    let model_config = ModelConfig::minimax_m25();

    // 1. Initialize substrate (weight storage)
    let data_dir = shellexpand::tilde(&cli.data_dir).to_string();
    let mut substrate = mycelium_substrate::SubstrateManager::new(&data_dir, model_config.clone());
    substrate.init().await?;
    substrate.scan().await?;
    info!("Substrate initialized at {}", data_dir);

    // 2. Initialize compute engine
    let device = candle_core::Device::Cpu;
    let mut engine = mycelium_compute::InferenceEngine::new(device);

    // 3. Load model if path provided
    let _model_loaded = if let Some(model_path) = &cli.model_path {
        info!("Loading model from: {}", model_path.display());
        match engine.load_model(model_path) {
            Ok(info) => {
                info!("Model loaded: {} tensors", info.tensor_count);
                true
            }
            Err(e) => {
                warn!("Failed to load model: {}. Running without model.", e);
                false
            }
        }
    } else {
        warn!("No model path provided. Node will run in coordinator-only mode.");
        false
    };

    // 4. Load tokenizer if provided
    if let Some(tok_path) = &cli.tokenizer_path {
        match engine.load_tokenizer(tok_path) {
            Ok(()) => info!("Tokenizer loaded from {}", tok_path.display()),
            Err(e) => warn!("Failed to load tokenizer: {}", e),
        }
    }

    // Wrap engine for the Fruit API
    let backend = ComputeBackend {
        engine: Arc::new(RwLock::new(engine)),
    };
    let mut inference_service = InferenceService::new();
    inference_service.set_backend(Arc::new(RwLock::new(Box::new(backend))));

    // 5. Initialize P2P network
    let hyphae_config = mycelium_hyphae::HyphaeConfig {
        listen_addr: cli.listen.clone(),
        bootstrap_peers: cli.bootstrap.iter().cloned().collect(),
        ..Default::default()
    };

    info!("Initializing hyphae (P2P network)...");
    let network = mycelium_hyphae::HyphaeNetwork::new(hyphae_config).await?;
    info!("Local peer ID: {}", network.local_peer_id());

    let handle = network.start().await?;
    let local_peer_id = handle.local_peer_id();
    info!("P2P network started, listening for peers");

    // 6. Initialize nucleus (self-tuning)
    let nucleus = Arc::new(Mutex::new(
        mycelium_nucleus::Nucleus::new(model_config.clone(), node_id.clone())
    ));
    info!(
        "Nucleus initialized (LoRA rank={})",
        mycelium_core::LORA_DEFAULT_RANK
    );

    // 7. Initialize spore propagator if spore mode enabled
    let caps = NodeCapabilities::auto_detect();
    let node_state = NodeCapacityState {
        node_id: node_id.clone(),
        available_vram_mb: caps.vram_mb,
        available_ram_mb: caps.ram_mb,
        layer_range: (0, model_config.num_layers),
        uptime_secs: 0,
        has_lora: false,
        lora_improvement: 0.0,
    };

    let spore_propagator: Option<Arc<Mutex<SporePropagator>>> = if cli.spore_mode {
        info!("Spore mode enabled");
        let propagator = SporePropagator::new(PropagationConfig::default(), node_state.clone());
        Some(Arc::new(Mutex::new(propagator)))
    } else {
        None
    };

    // 8. Start API server in a background task
    let api_port = cli.api_port;
    let fruit_config = FruitConfig {
        listen_addr: format!("0.0.0.0:{}", api_port),
        ..Default::default()
    };

    // Build AppState with references to all components
    let status = NodeStatus {
        node_id: node_id.clone(),
        status: "starting".into(),
        ..Default::default()
    };

    let app_state = AppState {
        node_id: node_id.clone(),
        status: Arc::new(RwLock::new(status)),
        inference_tx: None,
        inference_rx: Arc::new(RwLock::new(None)),
        training_steps: Arc::new(RwLock::new(0)),
        experience_size: Arc::new(RwLock::new(0)),
        running_loss: Arc::new(RwLock::new(0.0)),
        connected_peers: Arc::new(RwLock::new(0)),
        network_vram_mb: Arc::new(RwLock::new(0)),
        assigned_layers: Arc::new(RwLock::new("none".into())),
        assigned_experts: Arc::new(RwLock::new(Vec::new())),
        inference_service: Arc::new(RwLock::new(Some(inference_service))),
    };

    // Clone state for the server task
    let server_state = app_state.clone();
    let server_config = fruit_config.clone();

    let server_handle = tokio::spawn(async move {
        info!("Starting Fruit API server on {}", server_config.listen_addr);
        if let Err(e) = mycelium_fruit::serve(server_state, server_config).await {
            error!("API server error: {}", e);
        }
    });

    info!("Fruit API will be available at http://localhost:{}", api_port);

    // 9. Main event loop — integrate all components
    info!("Starting main event loop...");
    info!("Node ID: {}", node_id);
    info!("P2P listening on: {}", cli.listen);
    info!("Local peer ID: {}", local_peer_id);

    if cli.spore_mode {
        info!("Spore mode enabled — node will replicate when conditions are met");
    }

    // Shared references for the event loop
    let nucleus_clone = nucleus.clone();
    let propagator_clone = spore_propagator.clone();
    let app_state_clone = app_state.clone();

    // Spawn hyphae event processor
    let event_handle = tokio::spawn(async move {
        run_hyphae_event_loop(
            handle,
            nucleus_clone,
            propagator_clone,
            app_state_clone,
            model_config.clone(),
        )
        .await;
    });

    // 10. Spawn nucleus training loop
    let nucleus_train_clone = nucleus.clone();
    let training_handle = tokio::spawn(async move {
        run_nucleus_training_loop(nucleus_train_clone).await;
    });

    // 11. Spawn periodic status updater (reads from nucleus for training metrics)
    let status_state = app_state.clone();
    let status_nucleus = nucleus.clone();
    let status_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            let mut status = status_state.status.write().await;
            status.uptime_secs += 5;
            status.status = "running".into();

            // Update training metrics from nucleus
            let nuc = status_nucleus.lock().await;
            status_state.update_training_metrics(
                nuc.step(),
                nuc.experience_size(),
                nuc.running_loss(),
            ).await;
        }
    });

    // Wait for Ctrl+C
    tokio::signal::ctrl_c().await?;
    info!("Shutting down mycelium node...");

    // Graceful shutdown
    server_handle.abort();
    event_handle.abort();
    training_handle.abort();
    status_handle.abort();

    info!("Mycelium node stopped.");
    Ok(())
}

/// Process hyphae events and dispatch them to the appropriate components.
async fn run_hyphae_event_loop(
    handle: mycelium_hyphae::HyphaeHandle,
    _nucleus: Arc<Mutex<mycelium_nucleus::Nucleus>>,
    _propagator: Option<Arc<Mutex<SporePropagator>>>,
    app_state: AppState,
    _config: ModelConfig,
) {
    info!("Hyphae event processor started");

    loop {
        match handle.next_event().await {
            Some(mycelium_hyphae::HyphaeEvent::PeerJoined { peer_id, capabilities }) => {
                info!("Peer joined: {} (VRAM: {}MB)", peer_id, capabilities.vram_mb);

                // Update app state
                app_state
                    .update_network_status(
                        handle.connected_peers().await,
                        capabilities.vram_mb,
                    )
                    .await;
            }

            Some(mycelium_hyphae::HyphaeEvent::PeerLeft { peer_id }) => {
                warn!("Peer left: {}", peer_id);
            }

            Some(mycelium_hyphae::HyphaeEvent::Message { source, message }) => {
                debug!("Received message from {}", source);

                match message {
                    HyphaeMessage::GradientDelta { layer_idx, delta: _, version, node_id: grad_node } => {
                        info!(
                            "Received gradient delta for layer {} v{} from {}",
                            layer_idx, version, grad_node
                        );
                    }

                    HyphaeMessage::SporeAvailable { spore_id: _, model_name, shard_count, total_size_mb } => {
                        info!(
                            "Spore available from {}: {} ({} shards, {}MB)",
                            source, model_name, shard_count, total_size_mb
                        );
                    }

                    HyphaeMessage::SporeRequest { spore_id, requester } => {
                        info!("Peer {} requested spore {}", requester, spore_id);
                    }

                    HyphaeMessage::TopologyUpdate { map } => {
                        info!(
                            "Topology update: {} nodes, {}MB total VRAM",
                            map.nodes.len(),
                            map.total_vram_mb()
                        );
                        app_state
                            .update_network_status(map.nodes.len(), map.total_vram_mb())
                            .await;
                    }

                    HyphaeMessage::LatentDispatch { stream_id, layer_idx, latent: _ } => {
                        debug!(
                            "Received latent for stream {} at layer {}",
                            stream_id, layer_idx
                        );
                    }

                    HyphaeMessage::NodeAnnounce { node_id: ann_node, capabilities, listen_addr } => {
                        info!(
                            "Node {} announced: VRAM={}MB, addr={}",
                            ann_node, capabilities.vram_mb, listen_addr
                        );
                        app_state
                            .update_network_status(
                                handle.connected_peers().await,
                                capabilities.vram_mb,
                            )
                            .await;
                    }

                    HyphaeMessage::NodeDeparture { node_id: dep_node } => {
                        warn!("Node {} is departing", dep_node);
                    }

                    _ => {
                        debug!("Unhandled hyphae message from {}", source);
                    }
                }
            }

            Some(mycelium_hyphae::HyphaeEvent::ConnectionEstablished { peer_id }) => {
                debug!("Connection established: {}", peer_id);
                let peers = handle.connected_peers().await;
                *app_state.connected_peers.write().await = peers;
            }

            Some(mycelium_hyphae::HyphaeEvent::ConnectionLost { peer_id }) => {
                debug!("Connection lost: {}", peer_id);
                let peers = handle.connected_peers().await;
                *app_state.connected_peers.write().await = peers;
            }

            Some(mycelium_hyphae::HyphaeEvent::ListeningOn { address }) => {
                info!("Listening on {}", address);
            }

            Some(mycelium_hyphae::HyphaeEvent::TopologyChanged { map }) => {
                debug!("Topology changed: {} nodes", map.nodes.len());
                *app_state.network_vram_mb.write().await = map.total_vram_mb();
            }

            None => {
                info!("Hyphae event channel closed");
                break;
            }
        }
    }

    info!("Hyphae event processor stopped");
}

/// Run the nucleus training loop — periodically runs training on collected experience.
async fn run_nucleus_training_loop(nucleus: Arc<Mutex<mycelium_nucleus::Nucleus>>) {
    info!("Nucleus training loop started");

    let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

    loop {
        interval.tick().await;

        let mut nuc = nucleus.lock().await;

        // Run a training step on accumulated experience
        match nuc.train_step() {
            Ok(messages) => {
                if !messages.is_empty() {
                    debug!("Training step complete, produced {} gradient messages", messages.len());
                }
            }
            Err(e) => {
                debug!("Training step skipped: {}", e);
            }
        }
    }
}

async fn handle_command(command: Commands, cli: &Cli, node_id: &NodeId) -> Result<()> {
    match command {
        Commands::Generate { prompt, max_tokens, temperature } => {
            info!("Generating from: '{}' (max_tokens={}, temp={})", prompt, max_tokens, temperature);
            println!("Generating...");
            println!();

            let device = candle_core::Device::Cpu;
            let mut engine = mycelium_compute::InferenceEngine::new(device);

            // Load model if available
            if let Some(model_path) = &cli.model_path {
                match engine.load_model(model_path) {
                    Ok(_) => {
                        // Load tokenizer if available
                        if let Some(tok_path) = &cli.tokenizer_path {
                            if let Err(e) = engine.load_tokenizer(tok_path) {
                                println!("  [Tokenizer load error: {}]", e);
                                println!("  Prompt: {}", prompt);
                                return Ok(());
                            }
                        }

                        println!("  [Running inference with loaded model...]");
                        match engine.generate(&prompt, max_tokens, temperature) {
                            Ok(response) => {
                                println!("  Generated: {}", response.text);
                                println!("  Tokens: {} new", response.new_token_count);
                            }
                            Err(e) => {
                                println!("  Inference error: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("  [Failed to load model: {}]", e);
                        println!("  Prompt: {}", prompt);
                    }
                }
            } else {
                println!("  [No model loaded — use --model-path]");
                println!("  Prompt: {}", prompt);
            }
        }

        Commands::LatentExplore { prompt, layer, morph_with } => {
            info!("Exploring latent space at layer {} for: '{}'", layer, prompt);
            println!("Exploring latent space...");
            println!("  Prompt: {}", prompt);
            println!("  Layer: {}", layer);

            let device = candle_core::Device::Cpu;
            let mut engine = mycelium_compute::InferenceEngine::new(device);

            if let Some(model_path) = &cli.model_path {
                if engine.load_model(model_path).is_ok() {
                    println!("  [Running with latent extraction...]");
                    match engine.extract_latent(&prompt, layer) {
                        Ok(latent) => {
                            println!("  Latent extracted: dim={}, layer={}", latent.dim, latent.layer_idx);
                            println!("  First 8 values: {:?}", &latent.data[..8.min(latent.data.len())]);
                        }
                        Err(e) => println!("  Latent extraction error: {}", e),
                    }
                }
            }

            if let Some(morph) = morph_with {
                println!("  Morphing with: {}", morph);
                // Extract latent from second prompt and lerp
                if let Some(model_path) = &cli.model_path {
                    if engine.load_model(model_path).is_ok() {
                        match engine.extract_latent(&morph, layer) {
                            Ok(latent2) => {
                                // We need the first latent too — re-extract
                                match engine.extract_latent(&prompt, layer) {
                                    Ok(latent1) => {
                                        let morphed = latent1.lerp(&latent2, 0.5);
                                        println!("  Morphed latent: dim={}, similarity={:.4}",
                                            morphed.dim,
                                            latent1.cosine_similarity(&latent2));
                                        println!("  First 8 values: {:?}", &morphed.data[..8.min(morphed.data.len())]);
                                    }
                                    Err(e) => println!("  Latent extraction error: {}", e),
                                }
                            }
                            Err(e) => println!("  Second latent extraction error: {}", e),
                        }
                    }
                }
            }
        }

        Commands::Spore { action } => {
            info!("Spore action: {}", action);
            println!("Spore: {}", action);

            let config = ModelConfig::minimax_m25();
            let caps = NodeCapabilities::auto_detect();
            let node_state = NodeCapacityState {
                node_id: node_id.clone(),
                available_vram_mb: caps.vram_mb,
                available_ram_mb: caps.ram_mb,
                layer_range: (0, config.num_layers),
                uptime_secs: 0,
                has_lora: false,
                lora_improvement: 0.0,
            };
            let propagator = SporePropagator::new(PropagationConfig::default(), node_state);

            match action.as_str() {
                "create" => {
                    println!("  Creating spore from current model state...");

                    let builder = SporeBuilder::new(config.clone(), node_id.clone())
                        .layer_range(0, config.num_layers);

                    if let Some(model_path) = &cli.model_path {
                        // Use async runtime to build from GGUF
                        match builder.build_from_gguf(model_path, 4).await {
                            Ok(spore) => {
                                let genome_size = spore.genome.data.len();
                                println!("  Spore created: {} bytes (compressed)", genome_size);
                                println!("  Layers: {}-{}, Experts: {:?}", 0, config.num_layers, spore.expert_ids);
                            }
                            Err(e) => {
                                println!("  Spore creation failed: {}", e);
                            }
                        }
                    } else {
                        // Create spore from config only (no real genome)
                        let _genome = SporeGenome::new(
                            Vec::new(),
                            4,
                            0,
                        );
                        println!("  Spore config created (no weights): {} layers", config.num_layers);
                        println!("  Use --model-path to create a spore with actual weights");
                    }
                }
                "list" => {
                    println!("  Checking for spores in substrate...");
                    println!("  Node VRAM: {}MB, Available spores: {}, Received spores: {}",
                        caps.vram_mb,
                        propagator.available_spore_count(),
                        propagator.received_spore_count());
                }
                "germinate" => {
                    println!("  Looking for dormant spores to germinate...");
                    if caps.vram_mb >= 4096 {
                        println!("  Sufficient VRAM detected ({}MB)", caps.vram_mb);
                    } else {
                        println!("  Low VRAM ({}MB) — germination may be limited", caps.vram_mb);
                    }
                }
                _ => {
                    println!("  Unknown spore action: {}", action);
                    println!("  Available: create, list, germinate");
                }
            }
        }

        Commands::Status => {
            println!("Mycelium Node Status");
            println!("  Node ID: {}", node_id);
            println!("  Model: {}", if cli.model_path.is_some() { "configured" } else { "not loaded" });
            println!("  Tokenizer: {}", if cli.tokenizer_path.is_some() { "configured" } else { "not loaded" });
            println!("  P2P: enabled");
            println!("  LoRA rank: {}", mycelium_core::LORA_DEFAULT_RANK);

            let caps = NodeCapabilities::auto_detect();
            println!("  GPU: {:?}", caps.gpu_type);
            println!("  VRAM: {}MB", caps.vram_mb);
            println!("  CPU cores: {}", caps.compute_units);
            println!("  RAM: {}MB", caps.ram_mb);
        }
    }

    Ok(())
}
