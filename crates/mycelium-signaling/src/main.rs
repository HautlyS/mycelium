//! # Mycelium Signaling Server
//!
//! A lightweight WebSocket signaling server for WebRTC peer discovery.
//! This server facilitates the initial SDP offer/answer and ICE candidate
//! exchange between peers. Once the WebRTC connection is established,
//! all communication is direct P2P with no server involvement.
//!
//! ## Usage
//! ```bash
//! cargo run --bin mycelium-signaling -- --port 9001
//! ```
//!
//! ## Protocol
//! Clients send JSON messages:
//! - `{"type": "join", "peer_id": "..."}` — Join a room
//! - `{"type": "offer", "target": "...", "sdp": "..."}` — Send offer
//! - `{"type": "answer", "target": "...", "sdp": "..."}` — Send answer
//! - `{"type": "ice-candidate", "target": "...", "candidate": {...}}` — Send ICE
//! - `{"type": "leave"}` — Leave the room

use axum::{
    Router,
    extract::{
        Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::{RwLock, mpsc};
use tracing::{Level, info, warn};
use tracing_subscriber::FmtSubscriber;

/// A connected client.
struct Client {
    #[allow(dead_code)]
    peer_id: String,
    sender: mpsc::UnboundedSender<Message>,
}

/// Shared application state.
struct AppState {
    /// Map of peer_id -> Client
    clients: RwLock<HashMap<String, Client>>,
}

/// Incoming signaling message from client.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
enum SignalingMessage {
    /// Join the signaling room
    #[serde(rename = "join")]
    Join { peer_id: String },
    /// Send an SDP offer to a target peer
    #[serde(rename = "offer")]
    Offer { target: String, sdp: String },
    /// Send an SDP answer to a target peer
    #[serde(rename = "answer")]
    Answer { target: String, sdp: String },
    /// Send an ICE candidate to a target peer
    #[serde(rename = "ice-candidate")]
    IceCandidate {
        target: String,
        candidate: serde_json::Value,
    },
    /// Leave the signaling room
    #[serde(rename = "leave")]
    Leave,
    /// Ping to keep connection alive
    #[serde(rename = "ping")]
    Ping,
}

/// Outgoing signaling message to client.
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
enum OutgoingMessage {
    #[serde(rename = "offer")]
    Offer { from: String, sdp: String },
    #[serde(rename = "answer")]
    Answer { from: String, sdp: String },
    #[serde(rename = "ice-candidate")]
    IceCandidate {
        from: String,
        candidate: serde_json::Value,
    },
    #[serde(rename = "peer-joined")]
    PeerJoined { peer_id: String },
    #[serde(rename = "peer-left")]
    PeerLeft { peer_id: String },
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "error")]
    Error { message: String },
}

/// Query parameters for the WebSocket endpoint.
#[derive(Debug, Deserialize)]
struct WsQuery {
    peer_id: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI args manually for simplicity
    let port: u16 = std::env::args()
        .skip_while(|a| a != "--port")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(9001);

    let host: String = std::env::args()
        .skip_while(|a| a != "--host")
        .nth(1)
        .unwrap_or_else(|| "0.0.0.0".to_string());

    // Initialize tracing
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("Mycelium Signaling Server starting on {}:{}", host, port);

    let state = Arc::new(AppState {
        clients: RwLock::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health_handler))
        .route("/", get(root_handler))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    info!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> impl IntoResponse {
    "OK"
}

async fn root_handler() -> impl IntoResponse {
    (
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        serde_json::json!({
            "service": "mycelium-signaling",
            "version": env!("CARGO_PKG_VERSION"),
            "status": "running",
            "endpoints": {
                "websocket": "/ws?peer_id=YOUR_ID",
                "health": "/health"
            }
        })
        .to_string(),
    )
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WsQuery>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state, query.peer_id))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>, initial_peer_id: Option<String>) {
    let (mut ws_sender, mut ws_receiver) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

    // Task: forward messages from rx to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if ws_sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Track peer_id (may be set later via join message)
    let mut current_peer_id: Option<String> = initial_peer_id;

    // Handle incoming messages
    while let Some(Ok(msg)) = ws_receiver.next().await {
        match msg {
            Message::Text(text) => {
                match serde_json::from_str::<SignalingMessage>(&text) {
                    Ok(signal) => {
                        match signal {
                            SignalingMessage::Join { peer_id } => {
                                info!("Peer joined: {}", peer_id);

                                // Remove old entry if rejoining
                                if let Some(old_id) = &current_peer_id {
                                    state.clients.write().await.remove(old_id);
                                }

                                current_peer_id = Some(peer_id.clone());

                                state.clients.write().await.insert(
                                    peer_id.clone(),
                                    Client {
                                        peer_id: peer_id.clone(),
                                        sender: tx.clone(),
                                    },
                                );

                                // Broadcast peer joined to all others
                                let broadcast_msg = OutgoingMessage::PeerJoined {
                                    peer_id: peer_id.clone(),
                                };
                                if let Ok(json) = serde_json::to_string(&broadcast_msg) {
                                    broadcast_message(&state, &peer_id, Message::Text(json)).await;
                                }
                            }
                            SignalingMessage::Offer { target, sdp } => {
                                if let Some(from) = &current_peer_id {
                                    info!("Offer from {} to {}", from, target);
                                    let msg = OutgoingMessage::Offer {
                                        from: from.clone(),
                                        sdp,
                                    };
                                    send_to_peer(&state, &target, msg).await;
                                }
                            }
                            SignalingMessage::Answer { target, sdp } => {
                                if let Some(from) = &current_peer_id {
                                    info!("Answer from {} to {}", from, target);
                                    let msg = OutgoingMessage::Answer {
                                        from: from.clone(),
                                        sdp,
                                    };
                                    send_to_peer(&state, &target, msg).await;
                                }
                            }
                            SignalingMessage::IceCandidate { target, candidate } => {
                                if let Some(from) = &current_peer_id {
                                    let msg = OutgoingMessage::IceCandidate {
                                        from: from.clone(),
                                        candidate,
                                    };
                                    send_to_peer(&state, &target, msg).await;
                                }
                            }
                            SignalingMessage::Leave => {
                                info!("Peer leaving");
                                break;
                            }
                            SignalingMessage::Ping => {
                                let _ = tx.send(Message::Text(
                                    serde_json::to_string(&OutgoingMessage::Pong).unwrap(),
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse message: {}", e);
                        let _ = tx.send(Message::Text(
                            serde_json::to_string(&OutgoingMessage::Error {
                                message: format!("Invalid message: {}", e),
                            })
                            .unwrap(),
                        ));
                    }
                }
            }
            Message::Close(_) => {
                info!("WebSocket closed");
                break;
            }
            Message::Ping(data) => {
                let _ = tx.send(Message::Pong(data));
            }
            Message::Pong(_) => {}
            Message::Binary(_) => {
                warn!("Binary messages not supported");
            }
        }
    }

    // Cleanup: remove peer from registry
    if let Some(peer_id) = &current_peer_id {
        state.clients.write().await.remove(peer_id);
        info!("Peer removed from registry: {}", peer_id);

        // Notify others
        let leave_msg = OutgoingMessage::PeerLeft {
            peer_id: peer_id.clone(),
        };
        if let Ok(json) = serde_json::to_string(&leave_msg) {
            broadcast_message(&state, peer_id, Message::Text(json)).await;
        }
    }

    send_task.abort();
}

/// Send a message to a specific peer.
async fn send_to_peer(state: &AppState, target: &str, msg: OutgoingMessage) {
    if let Ok(json) = serde_json::to_string(&msg) {
        let clients = state.clients.read().await;
        if let Some(client) = clients.get(target) {
            let _ = client.sender.send(Message::Text(json));
        } else {
            warn!("Target peer not found: {}", target);
        }
    }
}

/// Broadcast a message to all peers except the sender.
async fn broadcast_message(state: &AppState, exclude: &str, msg: Message) {
    let clients = state.clients.read().await;
    for (peer_id, client) in clients.iter() {
        if peer_id != exclude {
            let _ = client.sender.send(msg.clone());
        }
    }
}
