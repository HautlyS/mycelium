# MYCELIUM — Development Roadmap

> *"The journey of a thousand nodes begins with a single spore."*

This document outlines the development roadmap for Mycelium, from current state to long-term vision.

---

## Table of Contents

1. [Vision & North Star](#1-vision--north-star)
2. [Current State (v0.2.0)](#2-current-state-v020)
3. [Near Term: v0.3.0 — Multi-Node](#3-near-term-v030--multi-node)
4. [Medium Term: v1.0 — Production Ready](#4-medium-term-v10--production-ready)
5. [Growth: v1.5 — Intelligent Routing](#5-growth-v15--intelligent-routing)
6. [Maturation: v2.0 — Autonomous Network](#6-maturation-v20--autonomous-network)
7. [Long Term: v3.0 — Ecosystem](#7-long-term-v30--ecosystem)
8. [Research Frontiers](#8-research-frontiers)
9. [Milestone Tracking](#9-milestone-tracking)

---

## 1. Vision & North Star

### 1.1 Ultimate Goal

**A fully decentralized, self-improving AI network that:**
- Runs across heterogeneous devices (phones to datacenters)
- Learns continuously from all participants
- Preserves user privacy by design
- Has no single point of failure or control
- Provides high-quality AI inference to anyone

### 1.2 Success Metrics

| Metric | Current | v1.0 Target | v3.0 Target |
|--------|---------|-------------|-------------|
| **Node Count** | 1-10 | 1,000 | 100,000+ |
| **Model Quality** | Base model only | +10% via LoRA | +30% via federated |
| **Inference Latency** | N/A | <200ms (regional) | <100ms (edge) |
| **Uptime** | N/A | 99.9% | 99.99% |
| **Privacy** | Gradients shared | DP-protected | Zero-knowledge |

---

## 2. Current State (v0.2.0)

**Status**: ✅ Foundation Complete
**Date**: April 2026

### 2.1 What Works

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| **P2P Networking** | ✅ Complete | 28 | Kademlia DHT, gossipsub, Noise encryption |
| **GGUF Loading** | ✅ Complete | 15 | candle-based, quantized model support |
| **MoE Router** | ✅ Complete | 12 | Network-aware expert selection |
| **Spore Protocol** | ✅ Complete | 24 | Packaging, verification, propagation |
| **Federated LoRA** | ✅ Complete | 18 | Gradient bridge, FedAvg aggregation |
| **WebGPU Compute** | ✅ Complete | 10 | Latent ops, CPU fallback |
| **REST API** | ✅ Complete | 8 | /generate, /latent, /status, /health |
| **WASM Target** | ✅ Complete | 5 | Browser integration, WebGPU |

**Total**: 138 tests passing

### 2.2 Known Limitations

- Single-node testing only (multi-node untested)
- CUDA path untested (no GPU in dev environment)
- WebGPU matmul shader incomplete
- No Byzantine fault tolerance
- Gradient sharing untested at scale
- Pipeline parallelism not fully implemented

### 2.3 Architecture

```
✅ mycelium-core      → Foundation types, config
✅ mycelium-compute   → Inference engine, MoE router
✅ mycelium-hyphae    → P2P networking
✅ mycelium-nucleus   → Federated LoRA
✅ mycelium-spore     → Self-replication
✅ mycelium-substrate → Weight storage
✅ mycelium-fruit     → REST/WebSocket API
✅ mycelium-node      → CLI binary
✅ mycelium-web       → WASM browser target
```

---

## 3. Near Term: v0.3.0 — Multi-Node

**Target**: Q2 2026 (2-3 months)
**Theme**: *"First connections"*

### 3.1 Objectives

1. **Multi-node testing** — Run 2-10 nodes in parallel
2. **Pipeline parallelism** — Distribute layers across nodes
3. **Integration tests** — Automated multi-node test suite
4. **Bootstrap infrastructure** — Reliable bootstrap nodes

### 3.2 Features

#### 3.2.1 Pipeline Parallelism (High Priority)

```rust
// TODO: Implement
pub struct PipelineExecutor {
    layer_assignments: HashMap<LayerId, NodeId>,
    latent_channels: HashMap<NodeId, LatentStream>,
    micro_batch_size: usize,
}

impl PipelineExecutor {
    pub async fn execute_pipeline(
        &self,
        input: LatentVector,
        layers: Vec<LayerId>,
    ) -> Result<LatentVector> {
        // Stream latents through layers distributed across nodes
        // Implement micro-batching for throughput
        // Handle node failures gracefully
    }
}
```

**Acceptance Criteria**:
- [ ] 4 nodes can process a request cooperatively
- [ ] Latency < 500ms for 64-layer model
- [ ] Node dropout doesn't crash the pipeline

#### 3.2.2 Multi-Node Integration Tests

```rust
// tests/pipeline_parallel.rs
#[tokio::test]
async fn test_four_node_pipeline() {
    let nodes = setup_network(4).await;

    let result = nodes[0]
        .generate("Hello, world!", GenerationConfig::default())
        .await;

    assert!(result.is_ok());
    assert!(!result.unwrap().text.is_empty());
}
```

**Acceptance Criteria**:
- [ ] Automated test with 2, 4, 8 nodes
- [ ] Test survives node dropout
- [ ] Test verifies output quality

#### 3.2.3 Bootstrap Node Program

**Goal**: Provide reliable bootstrap nodes for new users

**Implementation**:
- Deploy 3-5 bootstrap nodes on public infrastructure
- Document bootstrap addresses in repository
- Provide Docker compose for community bootstrap operators

**Acceptance Criteria**:
- [ ] New nodes can join via bootstrap
- [ ] Bootstrap nodes have 99.9% uptime
- [ ] Documentation for running bootstrap nodes

#### 3.2.4 Enhanced Logging & Monitoring

```rust
pub struct NodeMetrics {
    peer_count: usize,
    messages_sent: u64,
    messages_received: u64,
    inference_count: u64,
    gradient_contributions: u64,
    spores_propagated: u64,
    uptime_seconds: u64,
}
```

**Acceptance Criteria**:
- [ ] `/metrics` endpoint for Prometheus
- [ ] Grafana dashboard template
- [ ] Alert thresholds for anomalies

### 3.3 Research Tasks

- [ ] Measure actual latency in multi-node setup
- [ ] Profile network bandwidth requirements
- [ ] Test MoE routing accuracy

### 3.4 Success Criteria for v0.3.0

- [ ] 10-node test network runs for 24 hours
- [ ] Pipeline parallelism works across 4 nodes
- [ ] All integration tests pass
- [ ] Documentation updated for multi-node deployment

---

## 4. Medium Term: v1.0 — Production Ready

**Target**: Q3-Q4 2026 (6-12 months)
**Theme**: *"Ready for the world"*

### 4.1 Objectives

1. **Scale to 1,000 nodes** — Test at meaningful scale
2. **Hierarchical topology** — Cluster-based coordination
3. **Gradient security** — Differential privacy + anomaly detection
4. **Production hardening** — Error handling, monitoring, recovery

### 4.2 Features

#### 4.2.1 Hierarchical Gossipsub

**Current**: Flat topic structure
**Target**: Hierarchical topics for scale

```
mycelium/global → Announcements only
    ├── mycelium/region/us-east → Regional coordination
    │   └── mycelium/cluster/nyc-1 → Pipeline group
    └── mycelium/region/eu-west
        └── mycelium/cluster/london-1
```

**Acceptance Criteria**:
- [ ] 1,000 nodes with <1s message propagation
- [ ] Cluster isolation on failure
- [ ] Cross-cluster sync works correctly

#### 4.2.2 Differential Privacy

```rust
pub fn apply_dp_noise(
    gradient: &GradientDelta,
    epsilon: f64,
) -> GradientDelta {
    // Add calibrated Gaussian noise
    // Ensure (ε, δ)-differential privacy
}
```

**Acceptance Criteria**:
- [ ] Configurable privacy budget (ε)
- [ ] Privacy-utility tradeoff documented
- [ ] Default settings provide strong privacy

#### 4.2.3 Gradient Compression

```rust
// Top-k sparsification
pub fn compress_topk(
    gradient: &GradientDelta,
    k_percent: f64,
) -> SparseGradient {
    // Keep only top k% largest values
    // Achieve 10-100x compression
}
```

**Acceptance Criteria**:
- [ ] 90% sparsity with <5% quality loss
- [ ] Error compensation for discarded gradients
- [ ] Bandwidth reduction measured

#### 4.2.4 Super-Peer Architecture

**Goal**: Designate high-capacity nodes as hubs

```rust
pub fn evaluate_hub_candidate(node: &NodeInfo) -> HubScore {
    // VRAM, bandwidth, uptime, reliability
    // Automatic promotion/demotion
}
```

**Acceptance Criteria**:
- [ ] Hub nodes handle 10x more connections
- [ ] Hub selection is fair and transparent
- [ ] No single point of failure

#### 4.2.5 Expert Replication

**Goal**: Replicate popular experts for load balancing

```rust
pub struct ExpertPlacement {
    expert_id: ExpertId,
    primary: NodeId,
    replicas: Vec<NodeId>,
    load: AtomicU64,
}
```

**Acceptance Criteria**:
- [ ] Hot experts replicated across 3+ nodes
- [ ] Load balancing across replicas
- [ ] Consistent replica management

### 4.3 Infrastructure

| Component | Status | Target |
|-----------|--------|--------|
| **CI/CD** | Basic | Comprehensive (test matrix, benchmarks) |
| **Docker** | Working | Multi-arch (amd64, arm64) |
| **Monitoring** | None | Prometheus + Grafana |
| **Bootstrap Nodes** | Manual | Community program |
| **Documentation** | Good | Complete with tutorials |

### 4.4 Success Criteria for v1.0

- [ ] 1,000 nodes in production network
- [ ] Model quality improves over time (measurable)
- [ ] Network survives 10% node loss
- [ ] Sub-second inference latency (regional)
- [ ] Privacy guarantees documented and verified
- [ ] Community of 50+ contributors

---

## 5. Growth: v1.5 — Intelligent Routing

**Target**: 2027 (12-18 months)
**Theme**: *"Learning to learn"*

### 5.1 Objectives

1. **Predictive caching** — Anticipate expert needs
2. **Adaptive topology** — Self-optimizing network structure
3. **Multi-model support** — Run different models cooperatively
4. **Mobile clients** — Android/iOS native apps

### 5.2 Features

#### 5.2.1 Predictive Expert Caching

```rust
pub struct PredictiveCache {
    pattern_detector: PatternDetector,
    preload_queue: Vec<ExpertId>,
    hit_rate: f64,
}

// Analyze request patterns to predict next experts
// Pre-load predicted experts before they're needed
```

**Target**: 30-50% latency reduction

#### 5.2.2 Adaptive Topology

**Goal**: Network self-optimizes based on latency and bandwidth

```rust
pub struct AdaptiveTopology {
    latency_matrix: HashMap<(NodeId, NodeId), Duration>,
    bandwidth_matrix: HashMap<(NodeId, NodeId), Bandwidth>,
    cluster_assignments: HashMap<NodeId, ClusterId>,
}

// Automatically form low-latency clusters
// Rebalance when network conditions change
```

#### 5.2.3 Multi-Model Support

**Goal**: Support multiple models running simultaneously

```rust
pub struct MultiModelNode {
    models: HashMap<ModelId, ModelInstance>,
    router: ModelRouter,  // Route requests to appropriate model
    resource_allocator: ResourceAllocator,
}
```

**Supported Models**:
- MiniMax M2.5 (230B MoE) — Primary
- LLaMA 3.x variants — Secondary
- Specialized LoRA adapters — Per-domain

### 5.3 Mobile Support

| Platform | Status | Target |
|----------|--------|--------|
| **Android** | Documented | Native app with CPU inference |
| **iOS** | Documented | Native app with Metal inference |
| **React Native** | None | Cross-platform app (alternative) |

### 5.4 Success Criteria for v1.5

- [ ] Predictive cache hit rate >60%
- [ ] Topology adapts within 5 minutes of changes
- [ ] 3+ models running simultaneously
- [ ] Mobile apps in app stores
- [ ] 5,000+ nodes in network

---

## 6. Maturation: v2.0 — Autonomous Network

**Target**: 2027-2028 (18-24 months)
**Theme**: *"Self-sustaining"*

### 6.1 Objectives

1. **Byzantine fault tolerance** — Survive malicious nodes
2. **Autonomous governance** — Community-driven decisions
3. **Cross-region sync** — Global model coherence
4. **Edge computing** — CDN integration

### 6.2 Features

#### 6.2.1 Byzantine Fault Tolerance

**Target**: Tolerate <33% malicious nodes

```rust
// Krum aggregation algorithm
pub fn krum_aggregate(
    gradients: Vec<GradientDelta>,
    f: usize,  // Max malicious nodes
) -> GradientDelta {
    // Select gradient closest to consensus
    // Resistant to Byzantine failures
}
```

#### 6.2.2 Autonomous Governance

**Goal**: Community votes on protocol changes

```rust
pub struct GovernanceProposal {
    id: ProposalId,
    description: String,
    changes: Vec<ProtocolChange>,
    voting_period: Duration,
    votes: HashMap<NodeId, Vote>,
    required_quorum: f64,
}
```

**Voting Weight**: Based on compute contribution

#### 6.2.3 Edge Computing Integration

**Goal**: Deploy nodes at CDNs and ISPs

```
User → Edge Node (5ms) → Regional Hub → Global Network
       (cache, filter)   (coordinate)   (learn)
```

**Partnerships**: Cloudflare, Fastly, Akamai (hypothetical)

### 6.3 Research

- [ ] Formal verification of critical protocols
- [ ] Information-theoretic privacy bounds
- [ ] Optimal federated learning strategies
- [ ] Emergent behavior in large-scale networks

### 6.4 Success Criteria for v2.0

- [ ] 10,000+ nodes globally
- [ ] Tolerates 10% malicious nodes
- [ ] Governance process functional
- [ ] Edge nodes deployed at 3+ locations
- [ ] Self-sustaining contributor community
- [ ] Academic papers published

---

## 7. Long Term: v3.0 — Ecosystem

**Target**: 2028+ (24+ months)
**Theme**: *"Beyond imagination"*

### 7.1 Vision

**A federated ecosystem of AI networks:**

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Mycelium    │    │  Other       │    │  Enterprise  │
│  Network     │◄──►│  Networks    │◄──►│  Networks    │
│  (Public)    │    │  (Public)    │    │  (Private)   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 7.2 Features

#### 7.2.1 Cross-Network Bridges

**Goal**: Different AI networks can collaborate

```rust
pub struct NetworkBridge {
    source_network: NetworkId,
    target_network: NetworkId,
    protocol_adapter: ProtocolAdapter,
    translation_layer: TranslationLayer,
}
```

#### 7.2.2 Multimodal Support

**Goal**: Text, image, audio, video processing

```rust
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

pub struct MultimodalRouter {
    modality_experts: HashMap<Modality, Vec<ExpertId>>,
    cross_modality_attention: CrossModalAttention,
}
```

#### 7.2.3 Sovereign Clusters

**Goal**: Regulatory compliance via data residency

- EU cluster (GDPR compliant)
- Enterprise cluster (private data)
- Research cluster (open science)

### 7.3 Speculative Features

- **Quantum-resistant cryptography** — Post-quantum security
- **Neuromorphic computing** — Brain-inspired architectures
- **Interplanetary networking** — Delay-tolerant protocols
- **Consciousness research** — What emerges at scale?

### 7.4 Success Criteria for v3.0

- [ ] 100,000+ nodes globally
- [ ] Multiple models, multiple modalities
- [ ] Cross-network collaboration
- [ ] Network value rivals centralized AI
- [ ] Self-governing community
- [ ] Academic recognition

---

## 8. Research Frontiers

### 8.1 Open Research Questions

| Question | Importance | Difficulty | Timeline |
|----------|-----------|------------|----------|
| **Optimal MoE routing** | HIGH | MEDIUM | v1.0 |
| **Federated learning convergence** | HIGH | HIGH | v1.5 |
| **Byzantine gradient aggregation** | HIGH | HIGH | v2.0 |
| **Latent-space decoding** | MEDIUM | HIGH | v2.0 |
| **Emergent capabilities** | MEDIUM | UNKNOWN | v3.0 |

### 8.2 Collaboration Opportunities

- **Universities** — Federated learning, distributed systems
- **Research labs** — Privacy, security, cryptography
- **Open source** — candle, libp2p, wgpu communities
- **Industry** — Cloud providers, CDN companies

### 8.3 Funding Sources

- **Grants** — Protocol Labs, Ethereum Foundation, etc.
- **Sponsorships** — GitHub Sponsors, Open Collective
- **Bounties** — Feature-specific funding
- **Consulting** — Custom deployments (reinvest in project)

---

## 9. Milestone Tracking

### 9.1 Completed Milestones

| Milestone | Date | Status | Notes |
|-----------|------|--------|-------|
| Architecture design | 2026-04 | ✅ | Complete |
| Core types and config | 2026-04 | ✅ | Complete |
| P2P networking | 2026-04 | ✅ | Complete |
| GGUF loading | 2026-04 | ✅ | Complete |
| MoE routing | 2026-04 | ✅ | Complete |
| Spore protocol | 2026-04 | ✅ | Complete |
| Federated LoRA | 2026-04 | ✅ | Complete |
| WebGPU compute | 2026-04 | ✅ | Complete |
| REST API | 2026-04 | ✅ | Complete |
| WASM target | 2026-04 | ✅ | Complete |

### 9.2 Upcoming Milestones

| Milestone | Target | Status | Dependencies |
|-----------|--------|--------|--------------|
| Multi-node tests | v0.3.0 | 🚧 | Pipeline parallelism |
| Pipeline parallelism | v0.3.0 | 🚧 | Async streaming |
| Bootstrap program | v0.3.0 | ⏳ | Infrastructure |
| Hierarchical topics | v1.0 | ⏳ | Multi-node stability |
| Differential privacy | v1.0 | ⏳ | Gradient infrastructure |
| Gradient compression | v1.0 | ⏳ | Aggregation working |
| Expert replication | v1.5 | ⏳ | Load balancing |
| Predictive caching | v1.5 | ⏳ | Pattern detection |
| BFT aggregation | v2.0 | ⏳ | Research |
| Governance system | v2.0 | ⏳ | Community growth |

### 9.3 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|------------|
| **candle limitations** | MEDIUM | HIGH | Contribute upstream, fallback to alternatives |
| **libp2p scalability** | LOW | MEDIUM | Hierarchical topology, custom protocols |
| **Model quality degrades** | MEDIUM | HIGH | Rigorous testing, rollback capability |
| **Community doesn't grow** | MEDIUM | HIGH | Outreach, documentation, easy onboarding |
| **Security vulnerability** | MEDIUM | CRITICAL | Defense in depth, audits, rapid response |
| **Regulatory challenges** | LOW | HIGH | Legal review, sovereign clusters |

---

## Contributing to the Roadmap

This roadmap is a **living document**. We welcome input from the community:

1. **Open an issue** to suggest changes or additions
2. **Comment on existing milestones** to provide input
3. **Volunteer to work on milestones** — see CONTRIBUTING.md

**Priority is determined by**:
- Impact on network functionality
- Community interest
- Resource availability
- Dependencies and blockers

---

*The roadmap will be reviewed and updated quarterly based on progress, community input, and changing priorities.*

**Last Updated**: April 10, 2026
**Version**: v0.2.0
**Next Review**: July 2026

---

## See Also

- [SCALING.md](SCALING.md) — Scaling analysis and strategies
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [GOVERNANCE.md](GOVERNANCE.md) — Decision-making processes
- [PERFORMANCE.md](PERFORMANCE.md) — Benchmarks and optimization
