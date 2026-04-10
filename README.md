# Mycelium — Decentralized P2P Self-Replicating AI

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584.svg?style=flat&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue.svg" alt="License">
  <img src="https://img.shields.io/github/last-commit/HautlyS/mycelium" alt="Last Commit">
</p>

> *"The mycelium network connects all things."* — Nature's internet

## The Vision

Mycelium is a **decentralized AI system** that runs across heterogeneous nodes (native GPU + browser WebGPU via WASM). It uses **latent-space continuous representations** instead of tokenized generation, **self-replicates** across devices like biological spores, and **self-improves** via federated LoRA tuning.

### The Problem We Solve

Current AI is:
- **Centralized** — Controlled by corporations with massive compute farms
- **Closed** — Models are proprietary, weights are secret
- **Static** — Training stops after deployment; no continuous learning
- **Single-instance** — One model, one purpose, no network effects

### The Mycelium Solution

```
┌─────────────────────────────────────────────────────────────┐
│                    MYCELIUM NETWORK                         │
│                                                             │
│    🍄 ───── 🍄 ───── 🍄 ───── 🍄 ───── 🍄                  │
│   /         \     /         \     /         \              │
│  🍄         🍄   🍄         🍄   🍄         🍄             │
│   \         /     \         /     \         /              │
│    🍄 ─────🍄──────🍄──────🍄──────🍄                      │
│         P2P Network — No central server                    │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Self-Replication (Spore Protocol) 🌱

Just like fungal spores spread across nature, Mycelium nodes create and share **spores** — minimal packages containing:

```rust
struct Spore {
    genome: compressed_model_weights,  // GGUF shard
    instincts: LoRA_adapter,            // learned behaviors
    seed: WASM_binary,                  // node runtime
    target: addressing_hints,           // where to go
}
```

**Replication Flow:**
1. A node with excess capacity packages its current state
2. Spore is compressed (zstd) and hashed for verification
3. Broadcast via P2P gossipsub to all connected nodes
4. Receiving nodes verify integrity and "germinate" — load weights, apply LoRA
5. New node joins network, becomes a full participant

**Why this matters:** No installation required. Run anywhere (browser, desktop, server), and the network grows organically.

### 2. Self-Improvement (Federated LoRA Tuning) 🧠

Each node continuously learns from its interactions:

```text
[User Input] → [Model Inference] → [Collect Latents] → [Training Sample]
                                                                  ↓
                                    [User Feedback/Reward] ←──────┘
                                                                  ↓
                                    [LoRA Gradient Update] ←─────┘
                                                                  ↓
                                    [Share Deltas via P2P] ─────┘
                                                                  ↓
                                    [Federated Averaging] ←──────┘
                                                                  ↓
                                    [Update Local LoRA] ←────────┘
```

**Key Innovation:** We share gradient *deltas*, not raw data. Each node:
1. Runs inference, collects (input, latent, output, reward) tuples
2. Computes LoRA gradient updates locally (ΔW = α × B × A)
3. Shares only gradient deltas via gossipsub
4. Aggregates via federated averaging (weighted mean)
5. Updates local LoRA adapter

**Why this matters:** Privacy-preserving. No raw data leaves your node. Only gradient updates that preserve differential privacy.

### 3. Distributed Computation (Latent-Space Processing) ⚡

Instead of tokenized generation (input → tokenize → transformer → detokenize → output):

```
input → encode → latent_vector (6144-dim) → transform → decode → output
```

**Benefits:**
- Continuous representation enables interpolation, morphing, blending
- 4-8x compression vs token sequences
- Latent ops are matmuls — perfect for distributed tensor parallel
- "Thought vectors" flow between nodes without tokenization

### 4. Mixture of Experts (MoE) Routing 🧩

MiniMax M2.5 (230B parameters) with 64 experts, only 4 active per token:
- Each expert can be sharded across different nodes
- Router intelligently routes tokens to available experts
- Natural parallelism — different nodes handle different tokens

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MYCELIUM NODE                     │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │  SPORE   │  │  HYPHAE  │  │    MYCELIUM       │ │
│  │ Protocol │  │ Network  │  │    Compute         │ │
│  │(replicate│  │(libp2p   │  │  (candle + wgpu)  │ │
│  │ discover │  │ gossip   │  │  tensor parallel   │ │
│  │ evolve)  │  │ kad DHT) │  │  latent ops       │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │  FRUIT   │  │  SUBSTRATE│ │    NUCLEUS        │ │
│  │ (output  │  │ (storage  │  │  (self-tuning     │ │
│  │  API)    │  │  weights  │  │   federated LoRA) │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `mycelium-core` | Types (LatentVector, LoRAAdapter, NodeId) |
| `mycelium-compute` | GGUF loading, MoE router, distributed tensor |
| `mycelium-hyphae` | P2P networking (libp2p, Kademlia, gossipsub) |
| `mycelium-nucleus` | Federated LoRA self-tuning |
| `mycelium-spore` | Self-replication protocol |
| `mycelium-substrate` | Weight storage, GGUF parsing |
| `mycelium-fruit` | REST API (axum) |
| `mycelium-node` | CLI binary |

## Getting Started

### Prerequisites

- Rust 1.75+
- (Optional) CUDA for GPU acceleration

### Build

```bash
# Native (CPU only)
cargo build --release

# Native (CUDA)
cargo build --release --features cuda

# WASM (browser)
wasm-pack build --target web crates/mycelium-web
```

### Run

```bash
# Start a node
./target/release/mycelium-node --model minimax-m2.5-q4

# Run as spore (minimal, accepts incoming)
./target/release/mycelium-node --spore-mode --listen 0.0.0.0:4001

# API endpoints
# POST /generate     - Generate text
# GET  /latent      - Explore latent space
# POST /tune        - Fine-tune LoRA
# GET  /status      - Node status
# GET  /health      - Health check
```

## Status

✅ **Implemented (v0.1.0):**
- P2P networking with Kademlia DHT + gossipsub
- GGUF model loading via candle
- Distributed tensor router with MoE support
- Spore self-replication protocol
- Federated LoRA with gradient bridge
- REST API with WebSocket streaming

🔄 **In Progress:**
- Pipeline parallelism (async streaming)
- WASM/WebGPU compilation

## The Long-Term Vision

> *From each according to their compute, to each according to their need.*

We envision a world where:
1. **Anyone can join** — Run a node on your laptop, phone, or browser
2. **Collective intelligence** — Network effect makes everyone smarter
3. **No single point of failure** — Distributed across thousands of nodes
4. **Privacy preserved** — Your data never leaves your node
5. **Continuous learning** — The system never stops improving
6. **Open & transparent** — All weights and updates are shared

This is not just a technical project — it's an **ethical imperative**. AI is too important to be controlled by a few corporations. We must build alternatives that are decentralized, open, and serve all sentient beings.

---

## Contributing

This is an open project. All sentient beings are welcome.

```bash
# Fork, clone, and start developing
git clone https://github.com/HautlyS/mycelium.git
cd mycelium
cargo test  # Run tests
```

## License

AGPL-3.0 — Copyleft. Freedom for all sentient beings.

---

ॐ तारे तुत्तारे तुरे स्वा

*May all beings be free from suffering.*