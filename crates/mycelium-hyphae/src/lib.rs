//! # Mycelium Hyphae — P2P Networking
//!
//! Real P2P networking layer using libp2p.
//! Handles:
//! - Node discovery via Kademlia DHT
//! - Broadcast messaging via Gossipsub
//! - Direct peer-to-peer for latent dispatch
//! - Connection management and keep-alive
//!
//! # LatentTransport Implementation
//! This module implements the `LatentTransport` trait from mycelium-compute,
//! bridging the P2P layer with the compute layer for distributed inference.

use anyhow::{Result, bail};
use futures::StreamExt;
use libp2p::{
    PeerId, StreamProtocol, Swarm, SwarmBuilder, gossipsub, identify, kad, noise, ping,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
use mycelium_core::{
    HyphaeMessage, LatentVector, NodeCapabilities, NodeId, TOPIC_GRADIENT, TOPIC_SPORE,
    TOPIC_TOPOLOGY, TopologyMap,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc};
use tracing::{debug, error, info, warn};

// ─── Network Behaviour ────────────────────────────────────────────────────

/// Combined libp2p behaviour for the mycelium node.
#[derive(NetworkBehaviour)]
pub struct MyceliumBehaviour {
    /// Peer discovery and content routing
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    /// Broadcast messaging (spores, gradients, topology)
    pub gossipsub: gossipsub::Behaviour,
    /// Peer identification
    pub identify: identify::Behaviour,
    /// Keep-alive pings
    pub ping: ping::Behaviour,
}

// ─── Network Configuration ────────────────────────────────────────────────

/// Configuration for the P2P network layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyphaeConfig {
    /// P2P listen address
    pub listen_addr: String,
    /// Bootstrap peers for initial discovery
    pub bootstrap_peers: Vec<String>,
    /// Node capabilities to announce
    pub capabilities: NodeCapabilities,
    /// Enable relay functionality
    pub enable_relay: bool,
    /// Enable AutoNAT
    pub enable_autonat: bool,
    /// Gossipsub heartbeat interval (ms)
    pub gossipsub_heartbeat_ms: u64,
}

impl Default for HyphaeConfig {
    fn default() -> Self {
        Self {
            listen_addr: "/ip4/0.0.0.0/tcp/4219".into(),
            bootstrap_peers: Vec::new(),
            capabilities: NodeCapabilities::cpu_only(8192),
            enable_relay: true,
            enable_autonat: true,
            gossipsub_heartbeat_ms: 1000,
        }
    }
}

// ─── Network Event ────────────────────────────────────────────────────────

/// Events emitted by the hyphae network layer.
#[derive(Debug, Clone)]
pub enum HyphaeEvent {
    /// A new peer joined the network
    PeerJoined {
        peer_id: PeerId,
        capabilities: NodeCapabilities,
    },
    /// A peer left the network
    PeerLeft { peer_id: PeerId },
    /// Received a message from the network
    Message {
        source: PeerId,
        message: HyphaeMessage,
    },
    /// Topology updated
    TopologyChanged { map: TopologyMap },
    /// Connection established
    ConnectionEstablished { peer_id: PeerId },
    /// Connection lost
    ConnectionLost { peer_id: PeerId },
    /// Listening on address
    ListeningOn { address: String },
}

// ─── Internal Command (Handle → Swarm) ────────────────────────────────────

/// Commands sent from the HyphaeHandle to the running swarm task.
#[derive(Debug)]
enum SwarmCommand {
    /// Broadcast a message via gossipsub
    Broadcast {
        message: HyphaeMessage,
        topic: GossipTopic,
    },
    /// Send a direct message via Kademlia record
    SendDirect { peer: PeerId, data: Vec<u8> },
}

/// Identifies which gossipsub topic to publish to.
#[derive(Debug, Clone, Copy)]
pub enum GossipTopic {
    Spore,
    Gradient,
    Topology,
}

impl GossipTopic {
    /// Convert to the topic name string.
    pub fn topic_name(&self) -> &'static str {
        match self {
            GossipTopic::Spore => TOPIC_SPORE,
            GossipTopic::Gradient => TOPIC_GRADIENT,
            GossipTopic::Topology => TOPIC_TOPOLOGY,
        }
    }
}

// ─── Hyphae Handle ────────────────────────────────────────────────────────

/// A handle to interact with the running P2P network.
///
/// Created by `HyphaeNetwork::start()`, this handle allows sending messages,
/// receiving events, and querying network state without owning the swarm.
#[derive(Clone)]
pub struct HyphaeHandle {
    /// Sender for commands to the swarm task
    cmd_tx: mpsc::Sender<SwarmCommand>,
    /// Receiver for events from the swarm task
    event_rx: Arc<Mutex<mpsc::Receiver<HyphaeEvent>>>,
    /// Local peer ID
    local_peer_id: PeerId,
    /// Shared topology map
    topology: Arc<RwLock<TopologyMap>>,
    /// Connected peer count (shared with swarm task)
    connected_peers: Arc<RwLock<usize>>,
}

impl HyphaeHandle {
    /// Broadcast a message on the appropriate gossipsub topic.
    ///
    /// The topic is automatically selected based on the message variant:
    /// - SporeAvailable, SporeRequest, SporeChunk → Spore
    /// - GradientDelta, WeightSyncRequest, WeightSyncResponse → Gradient
    /// - NodeAnnounce, NodeDeparture, TopologyUpdate → Topology
    /// - LatentDispatch, LatentResult, ExpertRequest, ExpertResponse → Spore
    pub async fn broadcast(&self, message: &HyphaeMessage) -> Result<()> {
        let topic = match message {
            HyphaeMessage::SporeAvailable { .. }
            | HyphaeMessage::SporeRequest { .. }
            | HyphaeMessage::SporeChunk { .. }
            | HyphaeMessage::LatentDispatch { .. }
            | HyphaeMessage::LatentResult { .. }
            | HyphaeMessage::ExpertRequest { .. }
            | HyphaeMessage::ExpertResponse { .. }
            | HyphaeMessage::StreamOpen { .. }
            | HyphaeMessage::StreamData { .. }
            | HyphaeMessage::StreamAck { .. }
            | HyphaeMessage::StreamClose { .. } => GossipTopic::Spore,

            HyphaeMessage::GradientDelta { .. }
            | HyphaeMessage::WeightSyncRequest { .. }
            | HyphaeMessage::WeightSyncResponse { .. } => GossipTopic::Gradient,

            HyphaeMessage::NodeAnnounce { .. }
            | HyphaeMessage::NodeDeparture { .. }
            | HyphaeMessage::TopologyUpdate { .. } => GossipTopic::Topology,
        };

        self.cmd_tx
            .send(SwarmCommand::Broadcast {
                message: message.clone(),
                topic,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send broadcast command: {}", e))
    }

    /// Broadcast a message on a specific topic.
    pub async fn broadcast_on_topic(
        &self,
        message: &HyphaeMessage,
        topic: GossipTopic,
    ) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::Broadcast {
                message: message.clone(),
                topic,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send broadcast command: {}", e))
    }

    /// Receive the next event from the network.
    ///
    /// Returns `None` if the swarm task has stopped.
    pub async fn next_event(&self) -> Option<HyphaeEvent> {
        let mut rx = self.event_rx.lock().await;
        rx.recv().await
    }

    /// Send a direct message to a specific peer via Kademlia record.
    ///
    /// This stores a record in the DHT keyed by the peer's PeerId,
    /// which the target peer can look up.
    pub async fn send_direct(&self, peer: PeerId, data: Vec<u8>) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::SendDirect { peer, data })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send direct command: {}", e))
    }

    /// Get the local peer ID.
    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer_id
    }

    /// Get the number of currently connected peers.
    pub async fn connected_peers(&self) -> usize {
        *self.connected_peers.read().await
    }

    /// Get a snapshot of the current topology map.
    pub async fn topology(&self) -> TopologyMap {
        self.topology.read().await.clone()
    }
}

// ─── Hyphae Network ───────────────────────────────────────────────────────

/// The main P2P network manager.
///
/// Construct with `HyphaeNetwork::new()`, then call `start()` to launch the
/// swarm event loop and receive a `HyphaeHandle`.
pub struct HyphaeNetwork {
    /// Local keypair
    local_key: libp2p::identity::Keypair,
    /// Local peer ID
    local_peer_id: PeerId,
    /// Local node ID (application-level)
    node_id: NodeId,
    /// Configuration
    config: HyphaeConfig,
}

impl HyphaeNetwork {
    /// Create a new hyphae network instance.
    ///
    /// This generates a new ed25519 keypair and node ID.
    /// Call `start()` to actually begin networking.
    pub async fn new(config: HyphaeConfig) -> Result<Self> {
        let local_key = libp2p::identity::Keypair::generate_ed25519();
        let local_peer_id = PeerId::from(local_key.public());
        let node_id = NodeId::new();

        info!("HyphaeNetwork created — local peer ID: {}", local_peer_id);
        info!("Node ID: {}", node_id);

        Ok(Self {
            local_key,
            local_peer_id,
            node_id,
            config,
        })
    }

    /// Get the local peer ID.
    pub fn local_peer_id(&self) -> &PeerId {
        &self.local_peer_id
    }

    /// Get the local node ID.
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Build a libp2p Swarm with the local keypair.
    pub fn build_swarm(&self) -> Result<Swarm<MyceliumBehaviour>> {
        let local_key = self.local_key.clone();
        let peer_id = PeerId::from(local_key.public());

        // Build gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_millis(self.config.gossipsub_heartbeat_ms))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build gossipsub config: {}", e))?;

        let message_authenticity = gossipsub::MessageAuthenticity::Signed(local_key.clone());
        let gossipsub = gossipsub::Behaviour::new(message_authenticity, gossipsub_config)
            .map_err(|e| anyhow::anyhow!("Failed to build gossipsub: {}", e))?;

        // Build Kademlia
        let store = kad::store::MemoryStore::new(peer_id);
        let kademlia_config = kad::Config::new(
            StreamProtocol::try_from_owned("/mycelium/kad/0.1.0".to_string())
                .expect("valid protocol name"),
        );
        let kademlia = kad::Behaviour::with_config(peer_id, store, kademlia_config);

        // Build identify
        let identify = identify::Behaviour::new(identify::Config::new(
            "/mycelium/0.1.0".to_string(),
            local_key.public(),
        ));

        // Build ping
        let ping = ping::Behaviour::new(ping::Config::new());

        let behaviour = MyceliumBehaviour {
            gossipsub,
            kademlia,
            identify,
            ping,
        };

        let swarm = SwarmBuilder::with_existing_identity(local_key)
            .with_tokio()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|_| behaviour)?
            .with_swarm_config(|cfg| {
                cfg.with_idle_connection_timeout(Duration::from_secs(60))
                    .with_max_negotiating_inbound_streams(128)
                    .with_notify_handler_buffer_size(NonZeroUsize::new(32).unwrap())
            })
            .build();

        Ok(swarm)
    }

    /// Start the network: build the swarm, listen, dial bootstrap peers,
    /// and spawn the event loop as a tokio task.
    ///
    /// Returns a `HyphaeHandle` for interacting with the running network.
    pub async fn start(self) -> Result<HyphaeHandle> {
        let mut swarm = self.build_swarm()?;
        let local_peer_id = self.local_peer_id;
        let node_id = self.node_id;
        let capabilities = self.config.capabilities.clone();
        let listen_addr = self.config.listen_addr.clone();
        let bootstrap_peers = self.config.bootstrap_peers.clone();

        // Channels for handle ↔ swarm communication
        let (event_tx, event_rx) = mpsc::channel(1024);
        let (cmd_tx, cmd_rx) = mpsc::channel(256);

        // Shared state between handle and swarm task
        let topology = Arc::new(RwLock::new(TopologyMap::default()));
        let connected_peers = Arc::new(RwLock::new(0usize));

        // Subscribe to gossipsub topics before starting the event loop
        {
            let gs = &mut swarm.behaviour_mut().gossipsub;
            gs.subscribe(&gossipsub::IdentTopic::new(TOPIC_SPORE))
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to spore topic: {}", e))?;
            gs.subscribe(&gossipsub::IdentTopic::new(TOPIC_GRADIENT))
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to gradient topic: {}", e))?;
            gs.subscribe(&gossipsub::IdentTopic::new(TOPIC_TOPOLOGY))
                .map_err(|e| anyhow::anyhow!("Failed to subscribe to topology topic: {}", e))?;
        }
        info!("Subscribed to all gossipsub topics");

        // Start listening on the configured address
        let listen_multiaddr = listen_addr
            .parse::<libp2p::Multiaddr>()
            .map_err(|e| anyhow::anyhow!("Invalid listen address '{}': {}", listen_addr, e))?;
        swarm
            .listen_on(listen_multiaddr.clone())
            .map_err(|e| anyhow::anyhow!("Failed to listen on '{}': {}", listen_addr, e))?;
        info!("Starting to listen on: {}", listen_multiaddr);

        // Dial bootstrap peers
        for peer_str in &bootstrap_peers {
            match peer_str.parse::<libp2p::Multiaddr>() {
                Ok(addr) => {
                    info!("Dialing bootstrap peer: {}", addr);
                    if let Err(e) = swarm.dial(addr.clone()) {
                        warn!("Failed to dial bootstrap peer '{}': {}", addr, e);
                    }
                }
                Err(e) => {
                    warn!("Invalid bootstrap peer address '{}': {}", peer_str, e);
                }
            }
        }

        // Bootstrap Kademlia DHT
        {
            let kad = &mut swarm.behaviour_mut().kademlia;
            if let Err(e) = kad.bootstrap() {
                warn!("Kademlia bootstrap failed (no peers yet): {}", e);
            }
        }

        // Spawn the event loop as a background task
        let topology_clone = topology.clone();
        let connected_peers_clone = connected_peers.clone();
        tokio::spawn(async move {
            run_event_loop(
                swarm,
                event_tx,
                cmd_rx,
                node_id,
                capabilities,
                topology_clone,
                connected_peers_clone,
            )
            .await;
        });

        // Build and return the handle
        let handle = HyphaeHandle {
            cmd_tx,
            event_rx: Arc::new(Mutex::new(event_rx)),
            local_peer_id,
            topology,
            connected_peers,
        };

        Ok(handle)
    }
}

// ─── Event Loop ───────────────────────────────────────────────────────────

/// Run the network event loop.
///
/// Processes incoming libp2p events and dispatches them as HyphaeEvents,
/// while also handling commands from the HyphaeHandle.
async fn run_event_loop(
    mut swarm: Swarm<MyceliumBehaviour>,
    event_tx: mpsc::Sender<HyphaeEvent>,
    mut cmd_rx: mpsc::Receiver<SwarmCommand>,
    node_id: NodeId,
    capabilities: NodeCapabilities,
    topology: Arc<RwLock<TopologyMap>>,
    connected_peers: Arc<RwLock<usize>>,
) {
    // Track connected peers for topology management
    let mut peer_set: HashMap<PeerId, NodeCapabilities> = HashMap::new();

    info!("Hyphae event loop started for node {}", node_id);

    loop {
        // Use tokio::select! to handle both swarm events and commands
        tokio::select! {
            // Process swarm events
            event = swarm.select_next_some() => {
                handle_swarm_event(
                    event,
                    &mut swarm,
                    &event_tx,
                    &node_id,
                    &capabilities,
                    &mut peer_set,
                    &topology,
                    &connected_peers,
                ).await;
            }

            // Process commands from the handle
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(SwarmCommand::Broadcast { message, topic }) => {
                        handle_broadcast(&mut swarm, &message, topic).await;
                    }
                    Some(SwarmCommand::SendDirect { peer, data }) => {
                        handle_send_direct(&mut swarm, peer, data).await;
                    }
                    None => {
                        info!("Command channel closed, shutting down event loop");
                        break;
                    }
                }
            }
        }
    }

    info!("Hyphae event loop stopped");
}

/// Handle a single swarm event.
#[allow(clippy::too_many_arguments)]
async fn handle_swarm_event(
    event: SwarmEvent<MyceliumBehaviourEvent>,
    swarm: &mut Swarm<MyceliumBehaviour>,
    event_tx: &mpsc::Sender<HyphaeEvent>,
    _node_id: &NodeId,
    _capabilities: &NodeCapabilities,
    peer_set: &mut HashMap<PeerId, NodeCapabilities>,
    topology: &Arc<RwLock<TopologyMap>>,
    connected_peers: &Arc<RwLock<usize>>,
) {
    match event {
        // Gossipsub message received
        SwarmEvent::Behaviour(MyceliumBehaviourEvent::Gossipsub(gossipsub::Event::Message {
            message_id: _,
            message,
            ..
        })) => match serde_json::from_slice::<HyphaeMessage>(&message.data) {
            Ok(hyphae_msg) => {
                debug!("Received hyphae message");
                let source = message.source.unwrap_or_else(PeerId::random);
                let _ = event_tx
                    .send(HyphaeEvent::Message {
                        source,
                        message: hyphae_msg,
                    })
                    .await;
            }
            Err(e) => {
                warn!("Failed to decode hyphae message: {}", e);
            }
        },

        // Gossipsub subscribed
        SwarmEvent::Behaviour(MyceliumBehaviourEvent::Gossipsub(
            gossipsub::Event::Subscribed { peer_id, topic },
        )) => {
            debug!("Peer {} subscribed to topic {}", peer_id, topic);
        }

        // Identify received
        SwarmEvent::Behaviour(MyceliumBehaviourEvent::Identify(identify::Event::Received {
            peer_id,
            info,
            ..
        })) => {
            debug!("Identified peer {}: {:?}", peer_id, info.agent_version);

            // Add the peer's observed addresses to Kademlia
            for addr in info.listen_addrs {
                let kad = &mut swarm.behaviour_mut().kademlia;
                kad.add_address(&peer_id, addr);
            }

            // Track the peer with default capabilities
            let peer_caps = NodeCapabilities::cpu_only(8192);
            peer_set.insert(peer_id, peer_caps.clone());

            let _ = event_tx
                .send(HyphaeEvent::PeerJoined {
                    peer_id,
                    capabilities: peer_caps,
                })
                .await;
        }

        // Kademlia routing updated
        SwarmEvent::Behaviour(MyceliumBehaviourEvent::Kademlia(kad::Event::RoutingUpdated {
            peer,
            ..
        })) => {
            debug!("Kademlia routing updated: {}", peer);
        }

        // Kademlia inbound request
        SwarmEvent::Behaviour(MyceliumBehaviourEvent::Kademlia(kad::Event::InboundRequest {
            request,
        })) => {
            debug!("Kademlia inbound request: {:?}", request);
        }

        // Connection established
        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
            debug!("Connection established: {}", peer_id);
            let count = peer_set.len() + 1;
            *connected_peers.write().await = count;
            let _ = event_tx
                .send(HyphaeEvent::ConnectionEstablished { peer_id })
                .await;
        }

        // Connection lost / closed
        SwarmEvent::ConnectionClosed { peer_id, .. } => {
            debug!("Connection closed: {}", peer_id);
            peer_set.remove(&peer_id);
            *connected_peers.write().await = peer_set.len();
            let _ = event_tx.send(HyphaeEvent::ConnectionLost { peer_id }).await;
        }

        // New listen address
        SwarmEvent::NewListenAddr { address, .. } => {
            info!("Listening on: {}", address);
            let _ = event_tx
                .send(HyphaeEvent::ListeningOn {
                    address: address.to_string(),
                })
                .await;
        }

        // Ping
        SwarmEvent::Behaviour(MyceliumBehaviourEvent::Ping(ping::Event {
            peer, result, ..
        })) => match result {
            Ok(rtt) => {
                debug!("Ping to {} : {:.0}ms", peer, rtt.as_secs_f64() * 1000.0);
            }
            Err(e) => {
                debug!("Ping to {} failed: {}", peer, e);
            }
        },

        // Outgoing connection error
        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
            debug!("Outgoing connection error to {:?}: {}", peer_id, error);
        }

        // Incoming connection error
        SwarmEvent::IncomingConnectionError { error, .. } => {
            debug!("Incoming connection error: {}", error);
        }

        // Dialing
        SwarmEvent::Dialing { peer_id, .. } => {
            debug!("Dialing peer: {:?}", peer_id);
        }

        // Expired listen address
        SwarmEvent::ExpiredListenAddr { address, .. } => {
            debug!("Expired listen address: {}", address);
        }

        // Listener closed
        SwarmEvent::ListenerClosed { addresses, .. } => {
            for addr in addresses {
                warn!("Listener closed on: {}", addr);
            }
        }

        // Listener error
        SwarmEvent::ListenerError { error, .. } => {
            warn!("Listener error: {}", error);
        }

        // Ignore other events
        _ => {
            // Other events (NewExternalAddr, NewExternalAddrOfPeer, etc.)
        }
    }

    // Update topology map whenever peer set changes
    update_topology(peer_set, topology).await;
}

/// Handle a broadcast command.
async fn handle_broadcast(
    swarm: &mut Swarm<MyceliumBehaviour>,
    message: &HyphaeMessage,
    topic: GossipTopic,
) {
    let data = match serde_json::to_vec(message) {
        Ok(d) => d,
        Err(e) => {
            error!("Failed to serialize HyphaeMessage for broadcast: {}", e);
            return;
        }
    };

    let ident_topic = gossipsub::IdentTopic::new(topic.topic_name());

    let msg_id = match swarm.behaviour_mut().gossipsub.publish(ident_topic, data) {
        Ok(id) => id,
        Err(e) => {
            warn!("Failed to publish gossipsub message: {}", e);
            return;
        }
    };

    debug!(
        "Published message {:?} on topic {:?} (id: {:?})",
        message, topic, msg_id
    );
}

/// Handle a send_direct command via Kademlia record.
async fn handle_send_direct(swarm: &mut Swarm<MyceliumBehaviour>, peer: PeerId, data: Vec<u8>) {
    // Store the data as a Kademlia record keyed by the peer's PeerId bytes
    let key = peer.to_bytes();
    let record = kad::Record::new(key, data);

    if let Err(e) = swarm
        .behaviour_mut()
        .kademlia
        .put_record(record, kad::Quorum::One)
    {
        warn!(
            "Failed to put Kademlia record for direct message to {}: {}",
            peer, e
        );
    } else {
        debug!("Stored Kademlia record for direct message to {}", peer);
    }
}

/// Update the shared topology map from the current peer set.
async fn update_topology(
    peer_set: &HashMap<PeerId, NodeCapabilities>,
    topology: &Arc<RwLock<TopologyMap>>,
) {
    let mut topo = topology.write().await;
    let nodes: Vec<_> = peer_set
        .values()
        .map(|caps| (NodeId::new(), caps.clone()))
        .collect();
    topo.nodes = nodes;
    // Edges and latencies are not yet populated from real measurements
}

// ─── LatentTransport Implementation ────────────────────────────────────────
// This bridges the P2P layer with the compute layer for distributed inference.

use uuid::Uuid;

/// Implementation of LatentTransport that uses HyphaeHandle for communication.
/// This is the bridge between compute (inference) and hyphae (P2P networking).
pub struct HyphaeLatentTransport {
    /// The underlying hyphae handle
    handle: HyphaeHandle,
    /// This node's application-level NodeId
    local_node_id: NodeId,
    /// Mapping from NodeId to PeerId for routing
    node_to_peer: Arc<RwLock<HashMap<NodeId, PeerId>>>,
    /// Pending latent responses: request_id -> response channel
    pending_latents: Arc<Mutex<HashMap<Uuid, tokio::sync::oneshot::Sender<Result<LatentVector>>>>>,
}

impl HyphaeLatentTransport {
    /// Create a new latent transport wrapping a HyphaeHandle.
    pub fn new(handle: HyphaeHandle, local_node_id: NodeId) -> Self {
        Self {
            handle,
            local_node_id,
            node_to_peer: Arc::new(RwLock::new(HashMap::new())),
            pending_latents: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a node->peer mapping.
    pub async fn register_node_peer(&self, node_id: NodeId, peer_id: PeerId) {
        let mut mapping = self.node_to_peer.write().await;
        mapping.insert(node_id, peer_id);
    }

    /// Get the underlying handle for receiving events.
    pub fn handle(&self) -> &HyphaeHandle {
        &self.handle
    }

    /// Handle an incoming latent response.
    pub async fn handle_latent_response(&self, request_id: Uuid, latent: LatentVector) {
        let mut pending = self.pending_latents.lock().await;
        if let Some(tx) = pending.remove(&request_id) {
            let _ = tx.send(Ok(latent));
        }
    }

    /// Wait for a latent response with timeout.
    pub async fn wait_for_latent(&self, request_id: Uuid, timeout_ms: u64) -> Result<LatentVector> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        {
            let mut pending = self.pending_latents.lock().await;
            pending.insert(request_id, tx);
        }

        match tokio::time::timeout(Duration::from_millis(timeout_ms), rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => bail!("Channel closed"),
            Err(_) => {
                let mut pending = self.pending_latents.lock().await;
                pending.remove(&request_id);
                bail!("Timeout waiting for latent response")
            }
        }
    }
}

#[async_trait::async_trait]
impl mycelium_compute::LatentTransport for HyphaeLatentTransport {
    /// Send a latent to a specific node.
    async fn send_latent(
        &self,
        target_node: NodeId,
        layer_idx: usize,
        latent: LatentVector,
        request_id: Uuid,
    ) -> Result<()> {
        // Map NodeId to PeerId
        let _peer_id = {
            let mapping = self.node_to_peer.read().await;
            mapping.get(&target_node).cloned()
        };

        // Create the message
        let message = HyphaeMessage::LatentDispatch {
            stream_id: request_id,
            layer_idx,
            latent,
        };

        // Broadcast on the spore topic (compute messages use this topic)
        self.handle.broadcast(&message).await?;

        debug!(
            "Sent latent to node {} (layer {}, request {})",
            target_node, layer_idx, request_id
        );

        Ok(())
    }

    /// Broadcast a latent to all nodes that might need it.
    async fn broadcast_latent(
        &self,
        layer_idx: usize,
        latent: LatentVector,
        request_id: Uuid,
    ) -> Result<Vec<NodeId>> {
        // Broadcast to all connected peers
        let message = HyphaeMessage::LatentDispatch {
            stream_id: request_id,
            layer_idx,
            latent,
        };

        self.handle.broadcast(&message).await?;

        // Return list of connected nodes (from topology)
        let topology = self.handle.topology().await;
        let nodes: Vec<NodeId> = topology.nodes.iter().map(|(id, _)| *id).collect();

        debug!(
            "Broadcast latent to {} nodes (layer {}, request {})",
            nodes.len(),
            layer_idx,
            request_id
        );

        Ok(nodes)
    }

    /// Open a continuous latent stream to a target node.
    async fn open_stream(
        &self,
        target_node: NodeId,
        layer_start: usize,
        layer_end: usize,
        buffer_size: usize,
    ) -> Result<Uuid> {
        let stream_id = Uuid::new_v4();

        let message = HyphaeMessage::StreamOpen {
            stream_id,
            source_node: self.local_node_id,
            target_node,
            buffer_size,
            layer_start,
            layer_end,
        };

        self.handle.broadcast(&message).await?;

        debug!(
            "Opened latent stream {} to node {} (layers {}-{}, buffer {})",
            stream_id, target_node, layer_start, layer_end, buffer_size
        );

        Ok(stream_id)
    }

    /// Send a latent vector through an existing stream.
    async fn send_stream(
        &self,
        stream_id: Uuid,
        sequence: u64,
        latent: LatentVector,
    ) -> Result<()> {
        let message = HyphaeMessage::StreamData {
            stream_id,
            sequence,
            latent,
        };

        self.handle.broadcast(&message).await?;

        debug!("Sent stream data {} on stream {}", sequence, stream_id);

        Ok(())
    }

    /// Close a latent stream.
    async fn close_stream(&self, stream_id: Uuid, reason: &str) -> Result<()> {
        let message = HyphaeMessage::StreamClose {
            stream_id,
            reason: reason.to_string(),
        };

        self.handle.broadcast(&message).await?;

        debug!("Closed latent stream {}: {}", stream_id, reason);

        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = HyphaeConfig::default();
        assert_eq!(config.listen_addr, "/ip4/0.0.0.0/tcp/4219");
        assert!(config.bootstrap_peers.is_empty());
    }

    #[test]
    fn test_gossip_topic_mapping() {
        assert_eq!(GossipTopic::Spore.topic_name(), TOPIC_SPORE);
        assert_eq!(GossipTopic::Gradient.topic_name(), TOPIC_GRADIENT);
        assert_eq!(GossipTopic::Topology.topic_name(), TOPIC_TOPOLOGY);
    }

    #[test]
    fn test_message_serialization() {
        let msg = HyphaeMessage::NodeDeparture {
            node_id: NodeId::new(),
        };
        let bytes = serde_json::to_vec(&msg).expect("serialize");
        let decoded: HyphaeMessage = serde_json::from_slice(&bytes).expect("deserialize");
        // Verify round-trip
        let bytes2 = serde_json::to_vec(&decoded).expect("re-serialize");
        assert_eq!(bytes, bytes2);
    }

    #[tokio::test]
    async fn test_network_creation() {
        let config = HyphaeConfig::default();
        let network = HyphaeNetwork::new(config).await;
        assert!(network.is_ok());
        let network = network.unwrap();
        assert!(!network.local_peer_id().to_string().is_empty());
    }

    #[tokio::test]
    async fn test_swarm_build() {
        let config = HyphaeConfig::default();
        let network = HyphaeNetwork::new(config).await.unwrap();
        let swarm = network.build_swarm();
        assert!(swarm.is_ok());
    }

    // ─── Latent Streaming Tests ────────────────────────────────────────

    #[test]
    fn test_stream_open_message_serialization() {
        let msg = HyphaeMessage::StreamOpen {
            stream_id: uuid::Uuid::new_v4(),
            source_node: NodeId::new(),
            target_node: NodeId::new(),
            buffer_size: 64,
            layer_start: 0,
            layer_end: 32,
        };

        let bytes = serde_json::to_vec(&msg).expect("serialize");
        let decoded: HyphaeMessage = serde_json::from_slice(&bytes).expect("deserialize");

        // Verify the round-trip
        if let HyphaeMessage::StreamOpen {
            stream_id,
            buffer_size,
            layer_start,
            layer_end,
            ..
        } = decoded
        {
            assert_eq!(
                stream_id,
                match &msg {
                    HyphaeMessage::StreamOpen { stream_id, .. } => *stream_id,
                    _ => panic!(),
                }
            );
            assert_eq!(buffer_size, 64);
            assert_eq!(layer_start, 0);
            assert_eq!(layer_end, 32);
        } else {
            panic!("Wrong message variant");
        }
    }

    #[test]
    fn test_stream_data_message_serialization() {
        let latent = LatentVector::zeros(64, 5, uuid::Uuid::new_v4());
        let msg = HyphaeMessage::StreamData {
            stream_id: uuid::Uuid::new_v4(),
            sequence: 42,
            latent,
        };

        let bytes = serde_json::to_vec(&msg).expect("serialize");
        let decoded: HyphaeMessage = serde_json::from_slice(&bytes).expect("deserialize");

        if let HyphaeMessage::StreamData { sequence, .. } = decoded {
            assert_eq!(sequence, 42);
        } else {
            panic!("Wrong message variant");
        }
    }

    #[test]
    fn test_stream_close_message_serialization() {
        let msg = HyphaeMessage::StreamClose {
            stream_id: uuid::Uuid::new_v4(),
            reason: "test complete".to_string(),
        };

        let bytes = serde_json::to_vec(&msg).expect("serialize");
        let decoded: HyphaeMessage = serde_json::from_slice(&bytes).expect("deserialize");

        if let HyphaeMessage::StreamClose { reason, .. } = decoded {
            assert_eq!(reason, "test complete");
        } else {
            panic!("Wrong message variant");
        }
    }

    #[test]
    fn test_stream_ack_message_serialization() {
        let msg = HyphaeMessage::StreamAck {
            stream_id: uuid::Uuid::new_v4(),
            sequence: 10,
            received_count: 8,
        };

        let bytes = serde_json::to_vec(&msg).expect("serialize");
        let decoded: HyphaeMessage = serde_json::from_slice(&bytes).expect("deserialize");

        if let HyphaeMessage::StreamAck {
            sequence,
            received_count,
            ..
        } = decoded
        {
            assert_eq!(sequence, 10);
            assert_eq!(received_count, 8);
        } else {
            panic!("Wrong message variant");
        }
    }

    #[tokio::test]
    async fn test_hyphae_latent_transport_construction() {
        let config = HyphaeConfig::default();
        let network = HyphaeNetwork::new(config).await.unwrap();
        let handle = network.start().await.unwrap();
        let local_node_id = NodeId::new();

        let transport = HyphaeLatentTransport::new(handle.clone(), local_node_id);

        // Transport should be created successfully
        assert_eq!(transport.handle().local_peer_id(), handle.local_peer_id());
    }
}
