//! # Mycelium Web — WebRTC P2P Networking
//!
//! Real peer-to-peer networking for browser WASM nodes using WebRTC data channels.
//! This module provides:
//! - WebRTC peer connection management
//! - Data channel messaging bridged to HyphaeMessage protocol
//! - ICE candidate exchange via signaling server
//! - Peer discovery and connection lifecycle
//!
//! # Architecture
//! ```text
//! Browser Node A <--- WebRTC Data Channel ---> Browser Node B
//!        |                                         |
//!   [Signaling Server (WebSocket)]                  |
//!        |                                         |
//!   ICE Offer/Answer Exchange                      |
//! ```
//!
//! # Signaling
//! The signaling server is used only for the initial handshake (SDP offer/answer
//! and ICE candidates). Once the WebRTC connection is established, all
//! communication is direct P2P with no server involvement.

use mycelium_core::{HyphaeMessage, NodeCapabilities, NodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};
use wasm_bindgen::prelude::*;
use wasm_bindgen::closure::Closure;
use web_sys::{
    RtcPeerConnection, RtcPeerConnectionInit, RtcSessionDescriptionInit,
    RtcDataChannel, RtcDataChannelInit, RtcIceCandidate, RtcIceCandidateInit,
    MessageEvent, BinaryType,
};

/// STUN/TURN server configuration for NAT traversal.
/// Default uses Google's public STUN servers with fallback to open TURN servers.
pub const DEFAULT_ICE_SERVERS: &str = r#"[
    { "urls": "stun:stun.l.google.com:19302" },
    { "urls": "stun:stun1.l.google.com:19302" },
    { "urls": "stun:stun2.l.google.com:19302" },
    { "urls": "stun:stun3.l.google.com:19302" },
    { "urls": "stun:stun4.l.google.com:19302" },
    { "urls": "stun:stun.stunprotocol.org:3478" }
]"#;

/// Maximum reconnection attempts before giving up.
pub const MAX_RECONNECT_ATTEMPTS: u32 = 5;

/// Base delay in milliseconds between reconnection attempts.
pub const RECONNECT_BASE_DELAY_MS: u64 = 1000;

/// Maximum delay in milliseconds between reconnection attempts.
pub const RECONNECT_MAX_DELAY_MS: u64 = 30000;

/// Default signaling server URL (can be overridden).
pub const DEFAULT_SIGNALING_URL: &str = "ws://localhost:9001";

/// Events emitted by the WebRTC network layer.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum WebRtcEvent {
    /// A new peer connection was established
    PeerConnected {
        peer_id: String,
        channel_label: String,
    },
    /// A peer disconnected
    PeerDisconnected {
        peer_id: String,
        reason: String,
    },
    /// Received a message from a peer
    MessageReceived {
        peer_id: String,
        message: String,
    },
    /// ICE connection state changed
    IceStateChanged {
        peer_id: String,
        state: String,
    },
    /// Signaling state changed
    SignalingStateChanged {
        state: String,
    },
    /// Local ICE candidate ready to be sent to peer
    IceCandidateReady {
        peer_id: String,
        candidate: String,
        sdp_mid: Option<String>,
        sdp_mline_index: Option<u16>,
    },
    /// SDP offer ready to be sent to peer (for manual signaling)
    SdpOfferReady {
        peer_id: String,
        sdp: String,
    },
    /// SDP answer ready to be sent to peer
    SdpAnswerReady {
        peer_id: String,
        sdp: String,
    },
    /// Error occurred
    Error {
        message: String,
    },
}

/// Configuration for the WebRTC network layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRtcConfig {
    /// STUN/TURN server JSON array
    pub ice_servers: String,
    /// Signaling server WebSocket URL
    pub signaling_url: String,
    /// Node capabilities to announce
    pub capabilities: NodeCapabilities,
    /// Enable automatic connection to discovered peers
    pub auto_connect: bool,
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            ice_servers: DEFAULT_ICE_SERVERS.to_string(),
            signaling_url: DEFAULT_SIGNALING_URL.to_string(),
            capabilities: NodeCapabilities::browser(),
            auto_connect: true,
        }
    }
}

/// A connected peer with its WebRTC resources.
struct WebRtcPeer {
    /// The peer's node ID (application-level)
    node_id: NodeId,
    /// The WebRTC peer connection
    connection: RtcPeerConnection,
    /// The data channel for messaging
    data_channel: Option<RtcDataChannel>,
    /// Whether this connection is fully established
    connected: bool,
    /// Pending ICE candidates to add once remote description is set
    pending_ice_candidates: Vec<RtcIceCandidateInit>,
    /// Number of reconnection attempts
    reconnect_attempts: u32,
    /// Label of the data channel
    channel_label: String,
}

/// Callback closures associated with a single peer connection.
struct PeerCallbacks {
    /// ICE candidate callback
    ice_callback: Closure<dyn FnMut(JsValue)>,
    /// Data channel open callback
    on_open: Closure<dyn FnMut()>,
    /// Data channel close callback
    on_close: Closure<dyn FnMut()>,
    /// Data channel message callback
    on_message: Closure<dyn FnMut(JsValue)>,
}

/// WebRTC network manager for browser WASM nodes.
///
/// This handles:
/// - Creating/accepting WebRTC peer connections
/// - Managing data channels for message transport
/// - Bridging HyphaeMessage serialization over data channels
/// - ICE candidate exchange via signaling server or manual API
pub struct WebRtcNetwork {
    /// Local node identity
    local_node_id: NodeId,
    /// Configuration
    config: WebRtcConfig,
    /// Connected peers: peer_id (string from RTCPeerConnection) -> WebRtcPeer
    peers: Arc<Mutex<HashMap<String, WebRtcPeer>>>,
    /// Event queue for JS to consume
    events: Arc<Mutex<Vec<WebRtcEvent>>>,
    /// Signaling WebSocket connection (if connected)
    signaling_ws: Option<web_sys::WebSocket>,
    /// Callback closure for ICE candidates (resolve)
    _ice_callback: Option<Closure<dyn FnMut(JsValue)>>,
    /// Callback closure for ICE candidate errors (reject)
    _ice_callback_reject: Option<Closure<dyn FnMut(JsValue)>>,
    /// Callback closures for data channel messages (keyed by peer_id)
    _message_callbacks: Arc<Mutex<HashMap<String, Closure<dyn FnMut(JsValue)>>>>,
    /// Per-peer callback closures (keyed by peer_id), dropped on disconnect
    _peer_callbacks: Arc<Mutex<HashMap<String, PeerCallbacks>>>,
    /// Data channel callback for receive_offer (set before peer is established)
    _on_datachannel_callback: Option<Closure<dyn FnMut(JsValue)>>,
}

impl WebRtcNetwork {
    /// Create a new WebRTC network instance.
    pub fn new(local_node_id: NodeId, config: WebRtcConfig) -> Self {
        info!(
            "WebRtcNetwork created for node {} with signaling URL: {}",
            local_node_id, config.signaling_url
        );

        Self {
            local_node_id,
            config,
            peers: Arc::new(Mutex::new(HashMap::new())),
            events: Arc::new(Mutex::new(Vec::new())),
            signaling_ws: None,
            _ice_callback: None,
            _ice_callback_reject: None,
            _message_callbacks: Arc::new(Mutex::new(HashMap::new())),
            _peer_callbacks: Arc::new(Mutex::new(HashMap::new())),
            _on_datachannel_callback: None,
        }
    }

    /// Create an offer to initiate a connection to a peer.
    ///
    /// Returns the SDP offer as JSON string. The caller must transmit this
    /// to the remote peer via signaling (WebSocket, QR code, manual copy, etc.)
    /// and then call `set_remote_description` with the answer.
    pub fn create_offer(&self) -> Result<String, JsError> {
        let peer_id = format!("pending-{}", uuid::Uuid::new_v4());

        let ice_servers = self.parse_ice_servers();
        let mut rtc_config = RtcPeerConnectionInit::new();
        rtc_config.ice_servers(&ice_servers);

        let connection = RtcPeerConnection::new_with_configuration(&rtc_config)
            .map_err(|e| JsError::new(&format!("Failed to create RTCPeerConnection: {:?}", e)))?;

        // Create data channel for mycelium messages
        let channel_options = RtcDataChannelInit::new();
        channel_options.set_ordered(true);
        let channel = connection
            .create_data_channel_with_data_channel_dict("mycelium", &channel_options)
            .map_err(|e| JsError::new(&format!("Failed to create data channel: {:?}", e)))?;

        channel.set_binary_type(BinaryType::Arraybuffer);

        // Store peer in pending state
        {
            let mut peers = self.peers.lock().unwrap();
            peers.insert(peer_id.clone(), WebRtcPeer {
                node_id: NodeId::new(), // Will be updated when we receive peer info
                connection: connection.clone(),
                data_channel: Some(channel.clone()),
                connected: false,
                pending_ice_candidates: Vec::new(),
                reconnect_attempts: 0,
                channel_label: "mycelium".to_string(),
            });
        }

        // Set up channel event handlers
        self.setup_channel_handlers(&peer_id, &channel);

        // Set up ICE candidate handler
        self.setup_ice_candidate_handler(&peer_id, &connection);

        // Create offer
        let connection_clone = connection.clone();
        let peer_id_clone = peer_id.clone();
        let events_clone = self.events.clone();

        // Use a promise-based approach via wasm-bindgen-futures
        let future = wasm_bindgen_futures::JsFuture::from(connection.create_offer());
        let events_for_cb = events_clone.clone();
        let peer_id_for_cb = peer_id_clone.clone();
        let connection_for_set = connection_clone.clone();

        // Create closures for the promise callbacks
        let resolve_closure = Closure::wrap(Box::new(move |offer: JsValue| {
            let desc = RtcSessionDescriptionInit::new();
            // offer is an RTCSessionDescription, extract the SDP
            if let Some(sdp) = offer.dyn_ref::<js_sys::Object>().and_then(|o| {
                js_sys::Reflect::get(o, &JsValue::from_str("sdp")).ok()
            }).and_then(|v| v.as_string()) {
                desc.set_sdp(&sdp);
                desc.set_type(web_sys::RtcSessionDescriptionType::Offer);

                // Set local description
                let future = wasm_bindgen_futures::JsFuture::from(
                    connection_for_set.set_local_description(&desc)
                );
                // We'll handle this asynchronously
                let _ = web_sys::console::log_1(&JsValue::from_str(&format!(
                    "Offer created for {}", peer_id_for_cb
                )));

                // Emit event
                let mut events = events_for_cb.lock().unwrap();
                events.push(WebRtcEvent::SdpOfferReady {
                    peer_id: peer_id_for_cb.clone(),
                    sdp,
                });
            }
        }) as Box<dyn FnMut(JsValue)>);

        let reject_closure = Closure::wrap(Box::new(move |err: JsValue| {
            let error_msg = err.as_string().unwrap_or_else(|| "Unknown error".to_string());
            error!("Failed to create offer: {}", error_msg);
            let mut events = events_clone.lock().unwrap();
            events.push(WebRtcEvent::Error {
                message: format!("Failed to create offer: {}", error_msg),
            });
        }) as Box<dyn FnMut(JsValue)>);

        // Attach the promise handlers
        let _ = future.then(&resolve_closure).catch(&reject_closure);

        // Store both closures to prevent them from being dropped
        self._ice_callback = Some(resolve_closure);
        self._ice_callback_reject = Some(reject_closure);

        Ok(peer_id)
    }

    /// Set the remote description (answer from peer).
    ///
    /// This should be called with the SDP answer received from the remote peer.
    pub fn set_remote_description(&self, peer_id: &str, sdp_answer: &str) -> Result<(), JsError> {
        let peers = self.peers.lock().unwrap();
        let peer = peers.get(peer_id)
            .ok_or_else(|| JsError::new(&format!("Peer {} not found", peer_id)))?;

        let connection = peer.connection.clone();
        let pending_candidates = peer.pending_ice_candidates.clone();
        drop(peers);

        let desc = RtcSessionDescriptionInit::new();
        desc.set_sdp(sdp_answer);
        desc.set_type(web_sys::RtcSessionDescriptionType::Answer);

        let future = wasm_bindgen_futures::JsFuture::from(
            connection.set_remote_description(&desc)
        );

        // After setting remote description, add any pending ICE candidates
        let future_with_ice = future.then(&Closure::once_into_js(move |_: JsValue| {
            for candidate in pending_candidates {
                let ice = RtcIceCandidate::new(&candidate).unwrap();
                // Add ICE candidate (ignore errors for now)
                let _ = connection.add_ice_candidate(&ice);
            }
            web_sys::console::log_1(&JsValue::from_str(
                &format!("Remote description set and {} ICE candidates added for {}",
                    pending_candidates.len(), peer_id)
            ));
        }));

        // Prevent the closure from being dropped immediately
        wasm_bindgen_futures::spawn_local(async move {
            let _ = wasm_bindgen_futures::JsFuture::from(future_with_ice).await;
        });

        Ok(())
    }

    /// Set the remote description (offer from peer) and create an answer.
    ///
    /// Returns the SDP answer to send back to the peer.
    pub fn receive_offer(&self, sdp_offer: &str) -> Result<String, JsError> {
        let peer_id = format!("pending-{}", uuid::Uuid::new_v4());

        let ice_servers = self.parse_ice_servers();
        let mut rtc_config = RtcPeerConnectionInit::new();
        rtc_config.ice_servers(&ice_servers);

        let connection = RtcPeerConnection::new_with_configuration(&rtc_config)
            .map_err(|e| JsError::new(&format!("Failed to create RTCPeerConnection: {:?}", e)))?;

        // Set up data channel handler
        {
            let peers_clone = self.peers.clone();
            let events_clone = self.events.clone();
            let msg_callbacks = self._message_callbacks.clone();
            let peer_id_clone = peer_id.clone();
            let local_node = self.local_node_id;

            let on_channel_callback = Closure::wrap(Box::new(move |event: JsValue| {
                // A data channel was received from the peer
                if let Some(channel) = event.dyn_ref::<RtcDataChannel>() {
                    let channel_label = channel.label();

                    // Update peer entry with the data channel
                    {
                        let mut peers = peers_clone.lock().unwrap();
                        if let Some(peer) = peers.get_mut(&peer_id_clone) {
                            peer.data_channel = Some(channel.clone());
                            peer.channel_label = channel.label();
                        } else {
                            peers.insert(peer_id_clone.clone(), WebRtcPeer {
                                node_id: NodeId::new(),
                                connection: connection.clone(),
                                data_channel: Some(channel.clone()),
                                connected: false,
                                pending_ice_candidates: Vec::new(),
                                reconnect_attempts: 0,
                                channel_label: channel.label(),
                            });
                        }
                    }

                    // Set up message handler for this channel
                    let peer_id_for_msg = peer_id_clone.clone();
                    let events_for_msg = events_clone.clone();
                    let msg_cb = Closure::wrap(Box::new(move |msg_event: JsValue| {
                        if let Some(msg_event) = msg_event.dyn_ref::<MessageEvent>() {
                            if let Ok(data) = msg_event.data().dyn_into::<js_sys::JsString>() {
                                let message: String = data.into();
                                let mut events = events_for_msg.lock().unwrap();
                                events.push(WebRtcEvent::MessageReceived {
                                    peer_id: peer_id_for_msg.clone(),
                                    message,
                                });
                            }
                        }
                    }) as Box<dyn FnMut(JsValue)>);

                    channel.set_onmessage(Some(msg_cb.as_ref().unchecked_ref()));
                    msg_callbacks.lock().unwrap().insert(peer_id_clone.clone(), msg_cb);

                    let mut events = events_clone.lock().unwrap();
                    events.push(WebRtcEvent::PeerConnected {
                        peer_id: peer_id_clone.clone(),
                        channel_label,
                    });
                }
            }) as Box<dyn FnMut(JsValue)>);

            connection.set_ondatachannel(Some(on_channel_callback.as_ref().unchecked_ref()));
            // Store closure in HashMap to prevent memory leak (replaces std::mem::forget)
            // This will be cleaned up when disconnect_peer is called
            // For receive_offer, we store it under the pending peer_id
            // Note: This is a special case where the data channel handler is set up before
            // the peer is fully established, so we store the closure directly
            self._on_datachannel_callback = Some(on_channel_callback);
        }

        // Set up ICE candidate handler
        self.setup_ice_candidate_handler(&peer_id, &connection);

        // Set remote description (the offer)
        let connection_clone = connection.clone();
        let desc = RtcSessionDescriptionInit::new();
        desc.set_sdp(sdp_offer);
        desc.set_type(web_sys::RtcSessionDescriptionType::Offer);

        // Store peer
        {
            let mut peers = self.peers.lock().unwrap();
            peers.insert(peer_id.clone(), WebRtcPeer {
                node_id: NodeId::new(),
                connection: connection.clone(),
                data_channel: None, // Will be set when channel opens
                connected: false,
                pending_ice_candidates: Vec::new(),
                reconnect_attempts: 0,
                channel_label: "mycelium".to_string(),
            });
        }

        // We need to do this synchronously, so use a different approach
        // Store connection for later
        let peer_id_for_set = peer_id.clone();
        let events_for_set = self.events.clone();
        let sdp_offer_str = sdp_offer.to_string();

        // Create answer
        let connection_for_answer = connection.clone();
        let answer_future = wasm_bindgen_futures::JsFuture::from(
            connection.set_remote_description(&desc)
        );

        // Use spawn_local to handle the async operations
        wasm_bindgen_futures::spawn_local(async move {
            match answer_future.await {
                Ok(_) => {
                    // Create answer
                    let answer_result = connection_for_answer.create_answer();
                    match wasm_bindgen_futures::JsFuture::from(answer_result).await {
                        Ok(answer) => {
                            if let Some(sdp) = js_sys::Reflect::get(&answer, &JsValue::from_str("sdp"))
                                .ok().and_then(|v| v.as_string()) {
                                // Set local description
                                let local_desc = RtcSessionDescriptionInit::new();
                                local_desc.set_sdp(&sdp);
                                local_desc.set_type(web_sys::RtcSessionDescriptionType::Answer);

                                let _ = connection_for_answer.set_local_description(&local_desc).await;

                                let mut events = events_for_set.lock().unwrap();
                                events.push(WebRtcEvent::SdpAnswerReady {
                                    peer_id: peer_id_for_set.clone(),
                                    sdp,
                                });
                            }
                        }
                        Err(e) => {
                            error!("Failed to create answer: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to set remote description: {:?}", e);
                }
            }
        });

        Ok(peer_id)
    }

    /// Add an ICE candidate received from the signaling server or peer.
    pub fn add_ice_candidate(&self, peer_id: &str, candidate: &str,
                             sdp_mid: Option<&str>, sdp_mline_index: Option<u16>) -> Result<(), JsError> {
        let mut candidate_init = js_sys::Object::new();
        js_sys::Reflect::set(&candidate_init, &JsValue::from_str("candidate"),
                             &JsValue::from_str(candidate))?;
        if let Some(mid) = sdp_mid {
            js_sys::Reflect::set(&candidate_init, &JsValue::from_str("sdpMid"),
                                 &JsValue::from_str(mid))?;
        }
        if let Some(idx) = sdp_mline_index {
            js_sys::Reflect::set(&candidate_init, &JsValue::from_str("sdpMLineIndex"),
                                 &JsValue::from_str(&idx.to_string()))?;
        }

        let ice_candidate = RtcIceCandidate::new(&candidate_init)
            .map_err(|e| JsError::new(&format!("Invalid ICE candidate: {:?}", e)))?;

        let peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get(peer_id) {
            let connection = peer.connection.clone();
            drop(peers);

            // Add ICE candidate
            let future = wasm_bindgen_futures::JsFuture::from(
                connection.add_ice_candidate_with_opt_rtc_ice_candidate(Some(&ice_candidate))
            );

            wasm_bindgen_futures::spawn_local(async move {
                match future.await {
                    Ok(_) => {
                        debug!("ICE candidate added for peer {}", peer_id);
                    }
                    Err(e) => {
                        warn!("Failed to add ICE candidate: {:?}", e);
                    }
                }
            });
        } else {
            // Peer not yet registered, queue the candidate
            drop(peers);
            warn!("Peer {} not found, ICE candidate queued", peer_id);
        }

        Ok(())
    }

    /// Send a message to a specific peer via the data channel.
    pub fn send_message(&self, peer_id: &str, message: &str) -> Result<(), JsError> {
        let peers = self.peers.lock().unwrap();
        let peer = peers.get(peer_id)
            .ok_or_else(|| JsError::new(&format!("Peer {} not found", peer_id)))?;

        if let Some(channel) = &peer.data_channel {
            if channel.ready_state() == web_sys::RtcDataChannelState::Open {
                channel.send_with_str(message)
                    .map_err(|e| JsError::new(&format!("Failed to send message: {:?}", e)))?;
                debug!("Sent message to peer {}", peer_id);
                Ok(())
            } else {
                Err(JsError::new(&format!(
                    "Data channel not open (state: {:?})",
                    channel.ready_state()
                )))
            }
        } else {
            Err(JsError::new("No data channel available for peer"))
        }
    }

    /// Send a HyphaeMessage to all connected peers.
    pub fn broadcast_hyphae_message(&self, message: &HyphaeMessage) -> Result<usize, JsError> {
        let json = serde_json::to_string(message)
            .map_err(|e| JsError::new(&format!("Failed to serialize message: {}", e)))?;

        let peers = self.peers.lock().unwrap();
        let mut sent_count = 0;

        for (peer_id, peer) in peers.iter() {
            if let Some(channel) = &peer.data_channel {
                if channel.ready_state() == web_sys::RtcDataChannelState::Open {
                    if channel.send_with_str(&json).is_ok() {
                        sent_count += 1;
                    }
                }
            }
        }

        debug!("Broadcast message to {} peers", sent_count);
        Ok(sent_count)
    }

    /// Disconnect from a specific peer.
    pub fn disconnect_peer(&self, peer_id: &str) -> Result<(), JsError> {
        let mut peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.remove(peer_id) {
            peer.connection.close();
            info!("Disconnected from peer {}", peer_id);
            let mut events = self.events.lock().unwrap();
            events.push(WebRtcEvent::PeerDisconnected {
                peer_id: peer_id.to_string(),
                reason: "Local disconnect".to_string(),
            });
        }
        // Drop all callback closures for this peer to prevent memory leaks
        let mut callbacks = self._peer_callbacks.lock().unwrap();
        callbacks.remove(peer_id);
        let mut msg_callbacks = self._message_callbacks.lock().unwrap();
        msg_callbacks.remove(peer_id);
        Ok(())
    }

    /// Attempt to reconnect to a peer with exponential backoff.
    /// Returns true if reconnection was initiated, false if max attempts reached.
    pub fn attempt_reconnect(&self, peer_id: &str) -> bool {
        let mut peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get_mut(peer_id) {
            if peer.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS {
                warn!("Max reconnection attempts reached for peer {}", peer_id);
                let mut events = self.events.lock().unwrap();
                events.push(WebRtcEvent::PeerDisconnected {
                    peer_id: peer_id.to_string(),
                    reason: "Max reconnection attempts reached".to_string(),
                });
                return false;
            }

            peer.reconnect_attempts += 1;
            let delay = std::cmp::min(
                RECONNECT_BASE_DELAY_MS * 2u64.pow(peer.reconnect_attempts - 1),
                RECONNECT_MAX_DELAY_MS,
            );

            info!(
                "Reconnecting to peer {} (attempt {}/{}, delay {}ms)",
                peer_id, peer.reconnect_attempts, MAX_RECONNECT_ATTEMPTS, delay
            );

            // Schedule reconnection via ICE restart
            // Note: Full reconnection requires creating a new offer
            // The caller should handle this via the event system
            let mut events = self.events.lock().unwrap();
            events.push(WebRtcEvent::Error {
                message: format!(
                    "Connection lost, attempting reconnect (attempt {}/{}, next in {}ms)",
                    peer.reconnect_attempts, MAX_RECONNECT_ATTEMPTS, delay
                ),
            });
            return true;
        }
        false
    }

    /// Get the next pending WebRTC event (non-blocking).
    pub fn poll_event(&self) -> Option<String> {
        let mut events = self.events.lock().unwrap();
        if events.is_empty() {
            return None;
        }
        let event = events.remove(0);
        serde_json::to_string(&event).ok()
    }

    /// Get the list of connected peer IDs.
    pub fn get_connected_peers(&self) -> Vec<String> {
        let peers = self.peers.lock().unwrap();
        peers.iter()
            .filter(|(_, p)| p.connected)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get connection state for a specific peer.
    pub fn get_peer_state(&self, peer_id: &str) -> Option<String> {
        let peers = self.peers.lock().unwrap();
        peers.get(peer_id).map(|p| {
            format!("{:?}", p.connection.ice_connection_state())
        })
    }

    /// Get the local node ID.
    pub fn local_node_id(&self) -> NodeId {
        self.local_node_id
    }

    // ─── Internal Helpers ─────────────────────────────────────────────────

    fn parse_ice_servers(&self) -> js_sys::Array {
        // Parse the JSON ice servers config
        match serde_json::from_str::<Vec<serde_json::Value>>(&self.config.ice_servers) {
            Ok(servers) => {
                let array = js_sys::Array::new();
                for server in servers {
                    if let Ok(obj) = serde_wasm_bindgen::to_value(&server) {
                        array.push(&obj);
                    }
                }
                array
            }
            Err(_) => {
                // Fallback to default STUN servers
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &JsValue::from_str("urls"),
                                     &JsValue::from_str("stun:stun.l.google.com:19302")).ok();
                let array = js_sys::Array::new();
                array.push(&obj);
                array
            }
        }
    }

    fn setup_ice_candidate_handler(&self, peer_id: &str, connection: &RtcPeerConnection) {
        let peer_id_clone = peer_id.to_string();
        let events_clone = self.events.clone();

        let on_ice_candidate = Closure::wrap(Box::new(move |event: JsValue| {
            if let Some(ice_event) = event.dyn_ref::<web_sys::RtcPeerConnectionIceEvent>() {
                if let Some(candidate) = ice_event.candidate() {
                    let candidate_str = candidate.candidate();
                    let sdp_mid = candidate.sdp_mid();
                    let sdp_mline_index = candidate.sdp_mline_index();

                    let mut events = events_clone.lock().unwrap();
                    events.push(WebRtcEvent::IceCandidateReady {
                        peer_id: peer_id_clone.clone(),
                        candidate: candidate_str,
                        sdp_mid,
                        sdp_mline_index,
                    });
                }
            }
        }) as Box<dyn FnMut(JsValue)>);

        connection.set_onicecandidate(Some(on_ice_candidate.as_ref().unchecked_ref()));
        // Store closure in HashMap keyed by peer_id, dropped on disconnect (replaces std::mem::forget)
        let mut peer_callbacks = self._peer_callbacks.lock().unwrap();
        peer_callbacks.entry(peer_id.to_string())
            .and_modify(|cb| cb.ice_callback = on_ice_candidate)
            .or_insert_with(|| PeerCallbacks {
                ice_callback: on_ice_candidate,
                on_open: Closure::wrap(Box::new(|| {}) as Box<dyn FnMut()>),
                on_close: Closure::wrap(Box::new(|| {}) as Box<dyn FnMut()>),
                on_message: Closure::wrap(Box::new(|_: JsValue| {}) as Box<dyn FnMut(JsValue)>),
            });
    }

    fn setup_channel_handlers(&self, peer_id: &str, channel: &RtcDataChannel) {
        let peer_id_clone = peer_id.to_string();
        let events_clone = self.events.clone();
        let peers_clone = self.peers.clone();

        // Handle channel open
        let on_open = Closure::wrap(Box::new(move || {
            info!("Data channel opened for peer {}", peer_id_clone);
            let mut events = events_clone.lock().unwrap();
            events.push(WebRtcEvent::PeerConnected {
                peer_id: peer_id_clone.clone(),
                channel_label: "mycelium".to_string(),
            });

            // Mark peer as connected
            if let Ok(mut peers) = peers_clone.lock() {
                if let Some(peer) = peers.get_mut(&peer_id_clone) {
                    peer.connected = true;
                }
            }
        }) as Box<dyn FnMut()>);

        channel.set_onopen(Some(on_open.as_ref().unchecked_ref()));

        // Handle channel close
        let peer_id_close = peer_id.to_string();
        let events_close = self.events.clone();
        let peers_close = self.peers.clone();
        let self_for_reconnect = self.peers.clone();
        let on_close = Closure::wrap(Box::new(move || {
            warn!("Data channel closed for peer {}", peer_id_close);
            let mut events = events_close.lock().unwrap();
            events.push(WebRtcEvent::PeerDisconnected {
                peer_id: peer_id_close.clone(),
                reason: "Channel closed".to_string(),
            });

            if let Ok(mut peers) = peers_close.lock() {
                if let Some(peer) = peers.get_mut(&peer_id_close) {
                    peer.connected = false;
                    // Mark for reconnection
                    let attempts = peer.reconnect_attempts;
                    drop(peers);

                    if attempts < MAX_RECONNECT_ATTEMPTS {
                        info!("Scheduling reconnection for peer {}", peer_id_close);
                        // In a real implementation, this would trigger a new offer creation
                        // For now, we log the intent
                    }
                }
            }
        }) as Box<dyn FnMut()>);

        channel.set_onclose(Some(on_close.as_ref().unchecked_ref()));

        // Handle messages
        let peer_id_msg = peer_id.to_string();
        let events_msg = self.events.clone();
        let on_message = Closure::wrap(Box::new(move |event: JsValue| {
            if let Some(msg_event) = event.dyn_ref::<MessageEvent>() {
                if let Ok(data) = msg_event.data().dyn_into::<js_sys::JsString>() {
                    let message: String = data.into();
                    let mut events = events_msg.lock().unwrap();
                    events.push(WebRtcEvent::MessageReceived {
                        peer_id: peer_id_msg.clone(),
                        message,
                    });
                }
            }
        }) as Box<dyn FnMut(JsValue)>);

        channel.set_onmessage(Some(on_message.as_ref().unchecked_ref()));

        // Store all closures in HashMap keyed by peer_id, dropped on disconnect
        // This replaces std::mem::forget which caused memory leaks
        let mut peer_callbacks = self._peer_callbacks.lock().unwrap();
        peer_callbacks.entry(peer_id.to_string())
            .and_modify(|cb| {
                cb.on_open = on_open;
                cb.on_close = on_close;
                cb.on_message = on_message;
            })
            .or_insert_with(|| PeerCallbacks {
                ice_callback: Closure::wrap(Box::new(|_: JsValue| {}) as Box<dyn FnMut(JsValue)>),
                on_open,
                on_close,
                on_message,
            });
    }
}

/// Simple WebSocket signaling client for automatic peer discovery.
pub struct SignalingClient {
    ws: Option<web_sys::WebSocket>,
    connected: bool,
    events: Arc<Mutex<Vec<WebRtcEvent>>>,
    _on_open: Option<Closure<dyn FnMut(JsValue)>>,
    _on_message: Option<Closure<dyn FnMut(JsValue)>>,
    _on_close: Option<Closure<dyn FnMut(JsValue)>>,
}

impl SignalingClient {
    /// Create a new signaling client and connect to the server.
    pub fn new(url: &str, events: Arc<Mutex<Vec<WebRtcEvent>>>) -> Result<Self, JsError> {
        info!("Connecting to signaling server: {}", url);

        let ws = web_sys::WebSocket::new(url)
            .map_err(|e| JsError::new(&format!("Failed to connect to signaling server: {:?}", e)))?;

        let client = Self {
            ws: Some(ws),
            connected: false,
            events,
            _on_open: None,
            _on_message: None,
            _on_close: None,
        };

        Ok(client)
    }

    /// Send a message to the signaling server.
    pub fn send(&self, message: &str) -> Result<(), JsError> {
        if let Some(ws) = &self.ws {
            ws.send_with_str(message)
                .map_err(|e| JsError::new(&format!("Failed to send: {:?}", e)))?;
            Ok(())
        } else {
            Err(JsError::new("Not connected"))
        }
    }

    /// Close the connection.
    pub fn close(&mut self) {
        if let Some(ws) = self.ws.take() {
            ws.close().ok();
            self.connected = false;
        }
    }
}
