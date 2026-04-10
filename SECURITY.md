# MYCELIUM — Security Model & Threat Analysis

> *"Trust nothing. Verify everything. Assume some nodes are malicious."*

This document describes Mycelium's security architecture, threat model, attack vectors, and mitigation strategies.

---

## Table of Contents

1. [Security Philosophy](#1-security-philosophy)
2. [Threat Model](#2-threat-model)
3. [Cryptographic Primitives](#3-cryptographic-primitives)
4. [Network Security](#4-network-security)
5. [Weight Integrity](#5-weight-integrity)
6. [Federated Learning Security](#6-federated-learning-security)
7. [Spore Protocol Security](#7-spore-protocol-security)
8. [Compute Security](#8-compute-security)
9. [API Security](#9-api-security)
10. [Browser Security](#10-browser-security)
11. [Privacy Guarantees](#11-privacy-guarantees)
12. [Byzantine Fault Tolerance](#12-byzantine-fault-tolerance)
13. [Incident Response](#13-incident-response)
14. [Security Checklist](#14-security-checklist)
15. [Responsible Disclosure](#15-responsible-disclosure)

---

## 1. Security Philosophy

### 1.1 Core Principles

1. **Zero Trust** — No node is trusted by default. All messages are verified.
2. **Defense in Depth** — Multiple layers of security, so one failure doesn't compromise the system.
3. **Graceful Degradation** — Security failures affect individual nodes, not the entire network.
4. **Transparency** — All cryptographic operations are documented and auditable.
5. **Privacy by Design** — User data never leaves their node.

### 1.2 Trust Assumptions

| Component | Trust Assumption | Verification Method |
|-----------|-----------------|---------------------|
| **Peer Identity** | Untrusted | Ed25519 key verification |
| **Model Weights** | Untrusted | SHA-256 hash pinning |
| **Gradient Deltas** | Untrusted | Anomaly detection + DP |
| **Spore Contents** | Untrusted | CRC32 + SHA-256 + signature |
| **Routing Decisions** | Untrusted | Cryptographic receipts |

---

## 2. Threat Model

### 2.1 Adversary Capabilities

| Adversary Type | Resources | Goals |
|---------------|-----------|-------|
| **Casual Attacker** | Single node, basic tools | Disrupt service, spam network |
| **Malicious Node** | Multiple nodes, moderate compute | Poison model weights, degrade quality |
| **State-Level Actor** | Massive resources, many nodes | Censor content, deanonymize users |
| **Insider Threat** | Trusted contributor status | Subvert governance, introduce backdoors |

### 2.2 Attack Surface

```
┌─────────────────────────────────────────────────────────────┐
│                    ATTACK SURFACE                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────────┐ │
│  │  P2P     │  │  SPORE   │  │   FEDERATED LEARNING      │ │
│  │ Network  │  │ Protocol │  │   Gradient Poisoning      │ │
│  │ - Sybil  │  │ - Fake   │  │   - Model Poisoning       │ │
│  │ - Eclipse│  │ - Malware│  │   - Free Riding           │ │
│  │ - DoS    │  │ - Replay │  │   - Data Leakage          │ │
│  └──────────┘  └──────────┘  └───────────────────────────┘ │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────────┐ │
│  │  COMPUTE │  │   API    │  │   BROWSER (WASM)          │ │
│  │ - Side-ch│  │ - Auth   │  │   - Memory exhaustion     │ │
│  │ - Fault  │  │ - Rate   │  │   - Code injection        │ │
│  │ - Supply │  │ - CORS   │  │   - Data exfiltration     │ │
│  └──────────┘  └──────────┘  └───────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Risk Matrix

| Threat | Likelihood | Impact | Risk Level | Mitigation |
|--------|-----------|--------|------------|------------|
| **Sybil attack** | HIGH | MEDIUM | HIGH | Node reputation + stake |
| **Gradient poisoning** | MEDIUM | HIGH | HIGH | Anomaly detection + DP |
| **Weight substitution** | LOW | CRITICAL | MEDIUM | Hash pinning + signatures |
| **Eclipse attack** | MEDIUM | HIGH | HIGH | DHT diversity + relay |
| **DoS via spore flood** | MEDIUM | MEDIUM | MEDIUM | Rate limiting |
| **Memory exhaustion** | MEDIUM | MEDIUM | MEDIUM | Resource limits |
| **Supply chain attack** | LOW | CRITICAL | MEDIUM | Reproducible builds |

---

## 3. Cryptographic Primitives

### 3.1 Algorithms in Use

| Algorithm | Purpose | Parameters | Security Level |
|-----------|---------|------------|----------------|
| **Ed25519** | Node identity, message signing | 256-bit keys | 128-bit |
| **SHA-256** | Weight integrity, content hashing | 256-bit output | 128-bit |
| **Noise Protocol** | P2P connection encryption | X25519 + ChaCha20-Poly1305 | 128-bit |
| **CRC32** | Transport corruption detection | 32-bit | Error detection only |
| **HMAC-SHA256** | Future: authenticated channels | 256-bit key | 128-bit |

### 3.2 Key Management

```rust
/// Node identity with Ed25519 signing capability.
/// Keys are generated at first startup and never leave the node.
pub struct NodeIdentity {
    node_id: NodeId,               // Derived from public key
    signing_key: SigningKey,       // SECRET - never transmitted
    verifying_key: VerifyingKey,   // PUBLIC - shared during handshake
}
```

**Key Storage**:
- Private key: `$MYCELIUM_DATA_DIR/identity/ed25519.key`
- Permissions: `0600` (owner read/write only)
- Format: Raw 32-byte Ed25519 seed

**Key Rotation** (Future):
```rust
pub fn rotate_identity(old_identity: &NodeIdentity) -> NodeIdentity {
    // Generate new keypair
    // Sign new key with old key (chain of trust)
    // Broadcast key update via gossipsub
    // Old key remains valid for 24hr grace period
}
```

---

## 4. Network Security

### 4.1 P2P Connection Security

**Transport**: libp2p Noise protocol handshake

```
Node A                          Node B
   │                               │
   │──── X25519 public key ────────▶│
   │◀──── X25519 public key ───────│
   │                               │
   │── Noise_XX handshake ─────────▶│
   │   (mutual authentication)      │
   │                               │
   │◀── Encrypted channel ─────────│
   │   (ChaCha20-Poly1305)         │
```

**Security Properties**:
- Mutual authentication (both parties prove identity)
- Forward secrecy (compromised keys don't reveal past traffic)
- Man-in-the-middle resistance (X25519 key exchange)

### 4.2 Sybil Attack Mitigation

**Threat**: Attacker creates thousands of fake nodes to control the network.

**Mitigations**:

#### 4.2.1 Proof of Compute

```rust
pub struct ProofOfCompute {
    node_id: NodeId,
    challenge: [u8; 32],           // Random challenge from peers
    solution: [u8; 32],            // Hash of (challenge + compute_result)
    compute_result: LatentVector,  // Actual inference output
    timestamp: DateTime<Utc>,
}

impl ProofOfCompute {
    pub fn verify(&self, expected_output: &LatentVector) -> bool {
        // Verify the node actually performed computation
        self.compute_result == *expected_output
            && self.solution == sha256(&[challenge, compute_result])
    }
}
```

#### 4.2.2 Reputation System

```rust
pub struct NodeReputation {
    node_id: NodeId,
    uptime_score: f64,             // 0.0 - 1.0
    gradient_quality: f64,         // Measured against consensus
    spore_validity: f64,           // % of valid spores shared
    response_latency: f64,         // Average response time
    total_score: f64,              // Weighted combination
}
```

**Sybil Cost**: Each fake node must maintain:
- Stable uptime (costs bandwidth + compute)
- Valid gradient contributions (requires real model)
- Positive reputation (takes time to build)

### 4.3 Eclipse Attack Mitigation

**Threat**: Attacker controls all peers a victim node connects to.

**Mitigations**:
1. **DHT diversity** — Kademlia ensures connections to diverse peers
2. **Bootstrap redundancy** — Multiple bootstrap nodes (not controlled by single entity)
3. **Relay nodes** — libp2p relay provides alternate paths
4. **Peer scoring** — Detect and avoid suspicious peer clusters

```rust
// Detect potential eclipse: all peers have similar IDs
fn detect_eclipse_attempt(peers: &[PeerId]) -> bool {
    // Check ID distribution (should be uniform random)
    let id_prefixes: HashSet<_> = peers.iter()
        .map(|p| p.to_bytes()[0..4].to_vec())
        .collect();

    // If all peers share prefix, likely sybil/eclipse
    id_prefixes.len() < peers.len() / 2
}
```

### 4.4 DoS Mitigation

| Attack Vector | Mitigation |
|--------------|------------|
| **Spam messages** | Per-peer rate limiting (60s cooldown) |
| **Large spore floods** | Max 20GB per spore, 3 concurrent transfers |
| **Connection exhaustion** | Max 500 connections per node (configurable) |
| **CPU exhaustion** | Work rate limiting, async processing |
| **Memory exhaustion** | Bounded channels, backpressure |

```rust
// Rate limiter implementation
pub struct RateLimiter {
    peer_rates: HashMap<PeerId, Vec<Timestamp>>,
    max_events: usize,
    window_secs: Duration,
}

impl RateLimiter {
    pub fn allow(&mut self, peer: PeerId) -> bool {
        let now = Utc::now();
        let window_start = now - self.window_secs;

        // Remove old events
        self.peer_rates.entry(peer)
            .or_default()
            .retain(|t| *t > window_start);

        // Check rate
        let events = self.peer_rates.get_mut(&peer).unwrap();
        if events.len() >= self.max_events {
            return false;
        }

        events.push(now);
        true
    }
}
```

---

## 5. Weight Integrity

### 5.1 Weight Verification Registry

**Threat**: Malicious node substitutes model weights, causing incorrect outputs.

**Mitigation**: SHA-256 hash pinning (see CROSSDEVICE.md §13.5):

```rust
// Before loading any weights:
registry.verify(&model_name, &gguf_bytes, &source_peer)
    .map_err(|e| LoadError::WeightVerification(e))?;

// Only now is it safe to load
let model = candle_transformers::load_gguf(&gguf_bytes, &device)?;
```

**Registry Sources**:
1. **Local configuration** — User pins hashes manually
2. **Official releases** — Hashes published on GitHub releases
3. **Community consensus** — DHT-based governance proposals (future)

### 5.2 Supply Chain Security

**Threat**: Malicious code in dependencies or build process.

**Mitigations**:
1. **Cargo.lock** — Pinned dependency versions
2. **Reproducible builds** — Docker builds are deterministic
3. **Source audits** — All dependencies are open-source
4. **Minimal dependencies** — Only essential crates used

**Dependency Audit**:
```bash
# Check for known vulnerabilities
cargo audit

# Review dependency tree
cargo tree --depth 1

# Check for duplicate dependencies
cargo tree --duplicates
```

---

## 6. Federated Learning Security

### 6.1 Gradient Poisoning Attacks

**Threat**: Malicious node sends incorrect gradients to degrade model quality.

**Detection Methods**:

#### 6.1.1 Statistical Anomaly Detection

```rust
pub fn detect_gradient_anomaly(
    new_gradient: &GradientDelta,
    historical_gradients: &[GradientDelta],
    threshold_sigma: f64,  // Default: 3.0
) -> bool {
    // Compute mean and std of historical gradients
    let mean = historical_gradients.iter().mean();
    let std = historical_gradients.iter().std();

    // Check if new gradient is within threshold
    let z_score = (new_gradient.norm() - mean) / std;

    z_score.abs() < threshold_sigma
}
```

#### 6.1.2 Cosine Similarity Check

```rust
pub fn check_gradient_direction(
    new_gradient: &GradientDelta,
    consensus_direction: &Vector,
    min_similarity: f64,  // Default: 0.3
) -> bool {
    let similarity = cosine_similarity(new_gradient, consensus_direction);
    similarity > min_similarity
}
```

#### 6.1.3 Gradient Norm Clipping

```rust
// Prevent any single gradient from dominating
pub fn clip_gradient(gradient: &GradientDelta, max_norm: f64) -> GradientDelta {
    let current_norm = gradient.norm();
    if current_norm > max_norm {
        gradient * (max_norm / current_norm)
    } else {
        gradient.clone()
    }
}
```

### 6.2 Differential Privacy

**Threat**: Gradient deltas leak information about training data.

**Mitigation**: Add calibrated noise before sharing:

```rust
pub fn apply_differential_privacy(
    gradient: &GradientDelta,
    epsilon: f64,     // Privacy budget (ε)
    delta: f64,       // Failure probability (δ)
    sensitivity: f64, // Max gradient norm
) -> GradientDelta {
    // Gaussian mechanism
    let sigma = sensitivity * gaussian_mechanism_sigma(epsilon, delta);
    let noise = GaussianNoise::sample(sigma, gradient.len());

    gradient + noise
}
```

**Privacy Parameters**:
- `ε = 1.0`: Strong privacy, moderate utility loss
- `ε = 0.1`: Very strong privacy, significant utility loss
- `δ = 1e-5`: Standard failure probability

**See also**: [SCALING.md](SCALING.md#612-differential-privacy) for scaling implications of DP

### 6.3 Free Rider Detection

**Threat**: Node consumes model improvements without contributing.

**Detection**:
```rust
pub struct ContributionRecord {
    node_id: NodeId,
    gradients_shared: u64,
    inferences_served: u64,
    spores_propagated: u64,
    uptime_hours: f64,
    last_contribution: DateTime<Utc>,
}

impl ContributionRecord {
    pub fn is_free_rider(&self, threshold_hours: u64) -> bool {
        self.gradients_shared == 0
            && self.inferences_served == 0
            && self.spores_propagated == 0
            && self.uptime_hours > threshold_hours as f64
    }
}
```

**Mitigation**: Reduce update frequency for free riders (they still receive updates, just slower).

---

## 7. Spore Protocol Security

### 7.1 Spore Verification Chain

Every spore undergoes **triple verification** (see CROSSDEVICE.md §13.2):

```
1. CRC32 Check → Detects transport corruption (fast, not cryptographic)
      ↓ PASS
2. SHA-256 Hash → Verifies weight integrity (cryptographic)
      ↓ PASS
3. Ed25519 Signature → Authenticates author (non-repudiation)
      ↓ PASS
4. Weight Registry → Verifies against pinned hash
      ↓ PASS
5. Load into candle → Only now is it safe
```

**Failure at any step rejects the spore entirely**.

### 7.2 Spore DoS Protection

| Protection | Mechanism | Effect |
|-----------|-----------|--------|
| **Rate limiting** | 1 SporeAvailable per 60s per peer | Prevents announcement floods |
| **Size limits** | Max 20GB per spore | Bounds resource usage |
| **Transfer limits** | Max 3 concurrent inbound | Prevents memory exhaustion |
| **Chunk size** | Max 4MB per chunk | Enables streaming verification |

### 7.3 Malicious Spore Detection

**Threat**: Spore contains adversarial weights designed to:
- Produce specific outputs on trigger inputs (backdoor)
- Degrade overall model quality
- Exploit vulnerabilities in weight loader

**Mitigations**:
1. **Hash pinning** — Only accept known-good weights
2. **Sandboxed loading** — Future: load spores in isolated process
3. **Behavioral testing** — Future: test spore outputs on known inputs before acceptance

---

## 8. Compute Security

### 8.1 Side-Channel Mitigation

**Threat**: Timing attacks reveal information about model weights.

**Mitigations**:
- **Constant-time operations** — Where possible (challenging for ML)
- **Noise injection** — Add random delays to mask timing
- **Trusted execution** — Future: use SGX/TDX for sensitive operations

### 8.2 Fault Injection

**Threat**: Hardware faults cause incorrect computations.

**Mitigations**:
- **Redundant computation** — Future: verify critical ops on multiple nodes
- **Error-correcting codes** — For weight storage
- **Checksum verification** — On intermediate results

### 8.3 Model Extraction

**Threat**: Attacker queries API to reconstruct model weights.

**Difficulty**: High for 230B MoE model
- Requires ~45B queries (one per parameter)
- At 100 queries/sec: 1,400 years

**Mitigation**: Rate limiting on API endpoints

---

## 9. API Security

### 9.1 Authentication (Future)

**Current**: No authentication (open access)

**Future**: Optional API key authentication:
```rust
pub struct ApiAuth {
    api_key: ApiKey,
    rate_limit: RequestsPerMinute,
    permissions: Permissions,
}
```

### 9.2 Rate Limiting

```rust
pub struct ApiRateLimiter {
    per_ip: HashMap<IpAddr, Vec<Timestamp>>,
    per_key: HashMap<ApiKey, Vec<Timestamp>>,
    global_limit: usize,
    per_client_limit: usize,
    window: Duration,
}
```

**Default Limits**:
- 60 requests/minute per IP
- 1000 requests/minute per API key
- 10,000 requests/minute global

### 9.3 CORS Policy

```rust
// Restrict cross-origin access
let cors = CorsLayer::new()
    .allow_origin(AllowedOrigin::list([
        "https://hautlys.github.io".parse().unwrap(),
    ]))
    .allow_methods([Method::GET, Method::POST])
    .allow_headers([header::CONTENT_TYPE]);
```

---

## 10. Browser Security

### 10.1 WASM Sandboxing

**Browser nodes run in WASM sandbox**:
- No file system access
- No network access (except via WebGPU)
- Memory bounded by browser (2-4GB)
- No threads (unless SharedArrayBuffer enabled)

### 10.2 Memory Exhaustion

**Threat**: Malicious page allocates excessive WASM memory.

**Mitigation**: Browser enforces memory limits (2-4GB typical)

### 10.3 Code Injection

**Threat**: XSS or injection attacks via API responses.

**Mitigation**:
- All API responses are JSON (not HTML)
- Content-Security-Policy headers
- Input validation on all endpoints

---

## 11. Privacy Guarantees

### 11.1 Data Residency

**Guarantee**: User input data **never leaves their node**.

```
User Input → [Local Node] → [Local Inference] → Output
                      ↓
              [Local Gradient Computation]
                      ↓
              [Gradient Delta ONLY] → P2P Network
              (no raw data, no inputs, no outputs)
```

### 11.2 Metadata Privacy

**What IS visible to peers**:
- Node ID (pseudonymous)
- Compute capacity (VRAM, bandwidth)
- Gradient deltas (with DP noise)
- Uptime statistics

**What is NOT visible**:
- User inputs/prompts
- Model outputs/responses
- Raw training data
- Personal identifiers

### 11.3 Network-Level Privacy

**Current**: All P2P traffic encrypted via Noise protocol

**Future Enhancements**:
- Onion routing for gradient sharing
- Mix networks for metadata privacy
- Zero-knowledge proofs for contribution verification

---

## 12. Byzantine Fault Tolerance

### 12.1 Current Status

**v0.2.0**: Basic gradient aggregation (no BFT)

**Assumption**: Majority of nodes are honest

### 12.2 BFT Roadmap

| Version | BFT Level | Tolerance | Mechanism |
|---------|-----------|-----------|-----------|
| v0.2.0 | None | 0% | Trust all authenticated peers |
| v1.0 | Partial | <25% | Median-based aggregation |
| v2.0 | Strong | <33% | Krum/Multi-Krum algorithms |
| v3.0 | Full | <50% | BFT consensus + reputation |

### 12.3 Median-Based Aggregation (v1.0)

```rust
// Instead of mean (vulnerable to outliers), use median
pub fn median_aggregate(gradients: Vec<GradientDelta>) -> GradientDelta {
    let n = gradients.len();
    let mut sorted = gradients;
    sorted.sort_by_key(|g| g.norm());

    // Take median (or trimmed mean for smoother updates)
    let trimmed = &sorted[n / 4 .. 3 * n / 4];  // Remove top/bottom 25%
    trimmed.iter().mean()
}
```

---

## 13. Incident Response

### 13.1 Security Incident Types

| Incident | Severity | Response Time | Mitigation |
|----------|----------|---------------|------------|
| **Weight hash mismatch** | CRITICAL | Immediate | Reject spore, alert network |
| **Gradient anomaly spike** | HIGH | <1 hour | Quarantine node, review gradients |
| **DoS attack** | HIGH | <15 minutes | Rate limit, block attacker |
| **Key compromise** | CRITICAL | Immediate | Rotate key, revoke old key |
| **Vulnerability disclosure** | Variable | <48 hours | Patch, coordinate disclosure |

### 13.2 Node Quarantine

```rust
pub struct QuarantineManager {
    quarantined: HashMap<NodeId, QuarantineRecord>,
    auto_unban_after: Duration,
}

pub struct QuarantineRecord {
    node_id: NodeId,
    reason: QuarantineReason,
    timestamp: DateTime<Utc>,
    auto_release: DateTime<Utc>,
    appeal_count: u32,
}

enum QuarantineReason {
    GradientAnomaly,
    InvalidSpore,
    RateLimitViolation,
    ManualBan,
}
```

---

## 14. Security Checklist

### 14.1 Node Operator Checklist

- [ ] Generate unique Ed25519 keypair (automatic on first run)
- [ ] Verify weight hashes before loading models
- [ ] Enable firewall rules for P2P port (4001)
- [ ] Keep Mycelium updated (security patches)
- [ ] Monitor for anomalous gradient contributions
- [ ] Report suspicious behavior to network

### 14.2 Developer Checklist

- [ ] Run `cargo audit` before each release
- [ ] Review all dependency updates for security implications
- [ ] Test spore verification with malicious inputs
- [ ] Fuzz test P2P message handlers
- [ ] Review cryptographic implementations
- [ ] Update SECURITY.md with new threats

### 14.3 Release Checklist

- [ ] Reproducible Docker build
- [ ] SHA-256 hashes published in release notes
- [ ] Ed25519 signatures on release binaries
- [ ] Security audit summary included
- [ ] Upgrade path documented

---

## 15. Responsible Disclosure

### 15.1 Reporting Vulnerabilities

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. **Email**: security@mycelium.network (future)
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (optional)

### 15.2 Response Timeline

- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 1 week
- **Patch**: Within 2-4 weeks (depending on severity)
- **Disclosure**: Coordinated public disclosure after patch release

### 15.3 Bug Bounty (Future)

When funding permits:
- **Critical**: $5,000+
- **High**: $2,000-$5,000
- **Medium**: $500-$2,000
- **Low**: $100-$500

---

## Conclusion

Mycelium's security model is based on **zero trust, verify everything, assume some nodes are malicious**. We prioritize:

1. **Weight integrity** — Cryptographic verification before loading
2. **Gradient security** — Anomaly detection + differential privacy
3. **Network security** — Encrypted connections + rate limiting
4. **Privacy** — User data never leaves the node
5. **Resilience** — Network survives node compromises

**Security is a process, not a product**. We continuously audit, test, and improve.

---

*This document is updated as new threats are identified and mitigations are implemented.*

**Last Updated**: April 10, 2026
**Version**: v0.2.0
**Status**: Living document

---

## See Also

- [SCALING.md](SCALING.md) — Scaling analysis and strategies
- [GOVERNANCE.md](GOVERNANCE.md) — Network governance and decision-making
- [PERFORMANCE.md](PERFORMANCE.md) — Benchmarks and optimization
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues and solutions
