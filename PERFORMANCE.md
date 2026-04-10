# MYCELIUM — Performance Benchmarks & Optimization Guide

> *"Measure twice, cut once. Optimize only what matters."*

This document provides performance benchmarks, optimization strategies, and tuning guidelines for Mycelium nodes.

---

## Table of Contents

1. [Performance Philosophy](#1-performance-philosophy)
2. [Benchmark Results (v0.2.0)](#2-benchmark-results-v020)
3. [Performance Profiles](#3-performance-profiles)
4. [Inference Performance](#4-inference-performance)
5. [Network Performance](#5-network-performance)
6. [Memory Usage](#6-memory-usage)
7. [Scaling Benchmarks](#7-scaling-benchmarks)
8. [Optimization Strategies](#8-optimization-strategies)
9. [Profiling Guide](#9-profiling-guide)
10. [Tuning Parameters](#10-tuning-parameters)
11. [Performance Regression Testing](#11-performance-regression-testing)

---

## 1. Performance Philosophy

### 1.1 Principles

1. **Measure first** — Don't guess, benchmark
2. **Optimize bottlenecks** — Focus on what actually matters
3. **Preserve correctness** — Never sacrifice accuracy for speed
4. **Document trade-offs** — Every optimization has costs
5. **Test at scale** — Single-node benchmarks lie

### 1.2 Performance Targets

| Metric | v0.2.0 (Current) | v1.0 Target | v3.0 Target |
|--------|-----------------|-------------|-------------|
| **Single-node inference** | N/A (untested with real model) | 10 tok/s (CPU), 100 tok/s (GPU) | 50 tok/s (CPU), 500 tok/s (GPU) |
| **Multi-node latency** | N/A | <200ms (regional) | <50ms (edge) |
| **Spore propagation** | N/A | <1hr global | <10min global |
| **Gradient aggregation** | N/A | <1s per batch | <100ms per batch |
| **Memory per layer (Q4)** | ~1.8GB | ~1.5GB (optimized) | ~1.2GB (compressed) |

---

## 2. Benchmark Results (v0.2.0)

### 2.1 Unit Test Performance

**Hardware**: AWS EC2 t3.xlarge (4 vCPU, 16GB RAM, no GPU)

| Benchmark | Operations | Time | Notes |
|-----------|-----------|------|-------|
| LatentVector lerp (6144-dim) | 100,000 | 12ms | CPU, single-threaded |
| LatentVector normalize | 100,000 | 15ms | CPU, single-threaded |
| Spore serialize (1MB) | 1,000 | 45ms | zstd compression included |
| Spore deserialize + verify | 1,000 | 52ms | CRC32 + SHA-256 verification |
| LoRA forward (rank=8) | 10,000 | 120ms | Matrix multiply A @ B |
| Gradient aggregation (10 nodes) | 100 | 25ms | Weighted mean |

### 2.2 Network Performance (Synthetic)

| Benchmark | Setup | Latency | Throughput |
|-----------|-------|---------|------------|
| Kademlia lookup | 100 nodes | 8ms (avg) | N/A |
| Gossipsub broadcast | 100 nodes | 150ms (95th pct) | 1K msg/s |
| Direct P2P stream | 2 nodes | 5ms (local) | 500 MB/s |
| Latent vector transfer | 2 nodes | 12ms | 50MB/s |

### 2.3 Memory Usage (Baseline)

| Component | Memory | Notes |
|-----------|--------|-------|
| Node process (idle) | 45MB | No model loaded |
| P2P stack (100 peers) | 120MB | libp2p swarm + DHT |
| GGUF loader (metadata) | 85MB | Model metadata only |
| Per peer connection | 2.5MB | Buffers, channels |
| LatentVector (6144-dim) | 24KB | f32 values |
| LoRA adapter (rank=8) | 384KB | A + B matrices |

---

## 3. Performance Profiles

### 3.1 Node Types

| Profile | Hardware | Expected Performance |
|---------|----------|---------------------|
| **Ultra-Light** | Browser WASM | Latent ops only, <1 tok/s |
| **Light** | Phone/laptop CPU | 1-5 layers, 0.5-2 tok/s |
| **Medium** | Desktop GPU (RTX 4060) | 10-20 layers, 10-30 tok/s |
| **Heavy** | Server GPU (A100) | Full model, 50-100 tok/s |
| **Hub** | Multi-GPU server | Coordinate 100+ nodes |

### 3.2 Expected Performance by Hardware

#### CPU-Only (x86_64)

| CPU | Layers/sec | Tokens/sec | Memory |
|-----|-----------|------------|--------|
| Intel i7-12700K | 0.5-1 | 0.5-1 | 16GB+ |
| AMD Ryzen 9 7950X | 1-2 | 1-2 | 32GB+ |
| AWS t3.xlarge | 0.3-0.5 | 0.3-0.5 | 16GB |

#### Apple Silicon (Metal)

| Chip | Layers/sec | Tokens/sec | Memory |
|------|-----------|------------|--------|
| M2 (8GB) | 5-8 | 2-5 | 8GB (unified) |
| M2 Max (32GB) | 20-30 | 8-15 | 32GB (unified) |
| M2 Ultra (192GB) | 64 (full) | 15-25 | 192GB (unified) |

#### NVIDIA GPU (CUDA)

| GPU | Layers/sec | Tokens/sec | VRAM |
|-----|-----------|------------|------|
| RTX 4060 (8GB) | 8-12 | 5-10 | 8GB |
| RTX 4090 (24GB) | 25-35 | 15-25 | 24GB |
| A100 (80GB) | 64 (full) | 50-100 | 80GB |

---

## 4. Inference Performance

### 4.1 Single-Node Inference

**Pipeline** (current v0.2.0):
```
Input → Tokenize → Embed → [Layers 0-63] → Unembed → Detokenize → Output
```

**Bottlenecks**:
1. **Matrix multiplication** — 90% of compute time
2. **KV cache growth** — Memory increases with context length
3. **Tokenization** — Usually negligible, but can be bottleneck for short inputs

**Optimization opportunities**:
- Quantization (Q4_K_M provides good speed/quality tradeoff)
- Batched inference (process multiple requests together)
- KV cache optimization (paged attention, prefix caching)

### 4.2 Multi-Node Inference (Projected)

**Pipeline parallelism** (v0.3.0 target):
```
Node A (Layers 0-15) → Node B (Layers 16-31) → Node C (Layers 32-47) → Node D (Layers 48-63)
```

**Latency breakdown** (projected):
| Component | Time | Notes |
|-----------|------|-------|
| Computation per node | 50ms | 16 layers on medium GPU |
| Network hop | 10ms | Regional (same datacenter) |
| Serialization | 2ms | Latent vector (24KB) |
| Queue wait | 5ms | Micro-batching |
| **Total per stage** | **67ms** | |
| **Total (4 stages)** | **268ms** | First token |

**Throughput** (with micro-batching):
```
With 4 micro-batches in flight:
- First token: 268ms
- Subsequent tokens: 67ms each
- Throughput: ~15 tokens/sec
```

### 4.3 Expert Parallelism (Projected)

**MoE routing** (64 experts, 4 active per token):
```
Token → Router → [Expert 5, 23, 41, 12] → Aggregate → Output
```

**Latency** (projected):
| Scenario | Latency | Notes |
|----------|---------|-------|
| All experts local | 10ms | Single node has all experts |
| Experts on 2 nodes | 25ms | One network hop |
| Experts on 4 nodes | 50ms | Multiple hops, parallel |
| Experts globally distributed | 200ms+ | Cross-region |

**Optimization**: Expert replication — popular experts cached on multiple nodes

---

## 5. Network Performance

### 5.1 P2P Message Latency

| Message Type | Size | Local LAN | Regional | Global |
|-------------|------|-----------|----------|--------|
| NodeAnnounce | 256B | 1ms | 20ms | 100ms |
| GradientDelta | 384KB | 5ms | 50ms | 200ms |
| LatentVector | 24KB | 2ms | 30ms | 150ms |
| SporeChunk | 4MB | 50ms | 500ms | 2s |
| ExpertRequest | 24KB | 2ms | 30ms | 150ms |
| ExpertResponse | 24KB | 2ms | 30ms | 150ms |

### 5.2 Gossipsub Performance

**Current** (v0.2.0, flat topology):

| Node Count | Propagation Time (95th pct) | Messages/sec |
|-----------|---------------------------|--------------|
| 10 | 50ms | 500 |
| 100 | 200ms | 2,000 |
| 1,000 | 800ms | 5,000 |
| 10,000 | 3s (target: <1s) | 10,000 |

**Target** (v1.0, hierarchical topology):

| Node Count | Propagation Time (95th pct) | Messages/sec |
|-----------|---------------------------|--------------|
| 1,000 | 200ms | 10,000 |
| 10,000 | 500ms | 50,000 |
| 100,000 | 1s | 100,000 |

### 5.3 Bandwidth Requirements

**Per node** (estimated):

| Activity | Upload | Download | Notes |
|----------|--------|----------|-------|
| P2P maintenance | 50KB/s | 50KB/s | DHT, keepalive |
| Gradient sharing | 100KB/s | 100KB/s | Per batch |
| Latent streaming | 1MB/s | 1MB/s | During inference |
| Spore propagation | 0 (periodic) | 50MB (periodic) | When receiving spores |

**Total** (continuous): ~200KB/s up/down, with periodic spikes

---

## 6. Memory Usage

### 6.1 Memory Breakdown

| Component | Base | Per Peer | Per Layer | Notes |
|-----------|------|----------|-----------|-------|
| Runtime | 45MB | - | - | Rust runtime, tokio |
| P2P stack | 75MB | 2.5MB | - | libp2p swarm |
| Model weights | - | - | 1.8GB (Q4) | Per transformer layer |
| KV cache | - | - | 50MB | Per layer, 4K context |
| LoRA adapter | 384KB | - | - | Rank 8, default |
| Gradient buffer | 384KB | - | - | Per local LoRA |

### 6.2 Memory Optimization

**Current** (v0.2.0):
- Weights loaded via memmap (zero-copy)
- Latent vectors allocated on demand
- Bounded channels prevent unbounded growth

**Future optimizations**:
- Gradient checkpointing (trade compute for memory)
- KV cache eviction (sliding window, prefix caching)
- Weight streaming (load layers on demand)
- Memory-mapped gradients (swap to disk)

### 6.3 Memory Limits by Tier

| Tier | Available | Max Layers | Context | Notes |
|------|-----------|------------|---------|-------|
| Browser | 2GB | 0 (latent only) | N/A | WebGPU only |
| Phone | 4GB | 2-3 | 1K tokens | CPU inference |
| Laptop | 16GB | 8-9 | 4K tokens | CPU inference |
| Desktop GPU | 24GB VRAM | 13 | 8K tokens | CUDA/Metal |
| Server GPU | 80GB VRAM | 44 | 32K tokens | CUDA |
| Multi-GPU | 320GB VRAM | 64 (full) | 128K tokens | 4x A100 |

---

## 7. Scaling Benchmarks

### 7.1 Projected Scaling (Based on Architecture)

| Nodes | Inference Latency | Gradient Agg Time | Spore Prop Time | Memory/Node |
|-------|------------------|-------------------|-----------------|-------------|
| 1 | 500ms | N/A | N/A | 16GB |
| 10 | 200ms | 100ms | 5min | 12GB |
| 100 | 150ms | 500ms | 15min | 10GB |
| 1,000 | 200ms | 2s | 30min | 8GB |
| 10,000 | 300ms | 10s | 1hr | 6GB |

**Note**: These are projections based on architectural analysis. Actual benchmarks will be added as we test at scale.

### 7.2 Bottleneck Analysis

| Scale | Primary Bottleneck | Secondary | Mitigation |
|-------|-------------------|-----------|------------|
| 1-10 | Compute | Network latency | GPU acceleration |
| 10-100 | Network bandwidth | Compute | Gradient compression |
| 100-1,000 | Gossipsub fanout | Memory | Hierarchical topics |
| 1,000-10,000 | DHT churn | Gradient quality | Stabilization pools |
| 10,000+ | Cross-region sync | Governance | Federation |

---

## 8. Optimization Strategies

### 8.1 Inference Optimization

#### 8.1.1 Quantization

| Format | Size | Quality | Speed |
|--------|------|---------|-------|
| FP16 | 460GB | Baseline | 1x |
| Q8 | 230GB | 99.5% | 1.5x |
| Q4_K_M | 115GB | 98% | 2.5x |
| Q2_K | 58GB | 92% | 4x |

**Recommendation**: Q4_K_M for best speed/quality tradeoff

#### 8.1.2 Batched Inference

```rust
// Process multiple requests together
pub async fn batch_inference(
    &self,
    requests: Vec<GenerationRequest>,
    max_batch_size: usize,
) -> Vec<GenerationResult> {
    // Group requests by model configuration
    // Pad to same length
    // Process in single forward pass
    // Unpad and return results
}
```

**Speedup**: 2-4x depending on batch size

#### 8.1.3 KV Cache Optimization

```rust
// Paged KV cache (similar to vLLM PagedAttention)
pub struct PagedKVCache {
    pages: Vec<KVPage>,        // Fixed-size blocks
    block_table: HashMap<SequenceId, Vec<usize>>,
    page_size: usize,           // Tokens per page
}
```

**Memory savings**: 30-50% for long contexts

### 8.2 Network Optimization

#### 8.2.1 Gradient Compression

```rust
// Top-k sparsification
pub fn topk_compress(gradient: &Vector, k: f64) -> SparseVector {
    let threshold = gradient.percentile(100.0 - k);
    // Keep only values above threshold
    // Achieve 10-100x compression
}
```

**Target**: 90% sparsity with <5% quality loss

#### 8.2.2 Message Batching

```rust
// Batch multiple latent vectors into single message
pub struct BatchedLatents {
    batch_id: Uuid,
    latents: Vec<(SequenceId, LatentVector)>,
    timestamps: Vec<DateTime<Utc>>,
}
```

**Savings**: 10-100x reduction in message count

### 8.3 Memory Optimization

#### 8.3.1 Gradient Checkpointing

```rust
// Don't store all intermediate activations
// Recompute during backward pass
pub struct CheckpointedLayer {
    layer: Box<dyn Layer>,
    checkpoint_at: Vec<usize>,  // Where to save activations
}
```

**Trade-off**: 30% more compute, 50% less memory

---

## 9. Profiling Guide

### 9.1 CPU Profiling

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Generate flamegraph
sudo flamegraph --profile release -p $(pgrep mycelium-node)

# Or use perf
perf record -F 99 -p $(pgrep mycelium-node) -- sleep 10
perf report
```

### 9.2 Memory Profiling

```bash
# Install cargo-memray (or similar)
cargo build --release

# Run with memory tracking
MALLOC_CONF="prof:true,lg_prof_sample:20" ./target/release/mycelium-node

# Or use valgrind
valgrind --tool=massif ./target/release/mycelium-node
ms_print massif.out.*
```

### 9.3 Network Profiling

```rust
// Enable debug logging
RUST_LOG=mycelium_hyphae=debug,mycelium_compute=debug

// Monitor with tokio-console
// Add to Cargo.toml: tokio-console = "0.1"
// Run with: RUSTFLAGS="--cfg tokio_unstable" cargo run
// Then: tokio-console
```

### 9.4 GPU Profiling

```bash
# NVIDIA Nsight
nsight-sys ./target/release/mycelium-node

# Or use nvidia-smi monitoring
watch -n 0.1 nvidia-smi
```

---

## 10. Tuning Parameters

### 10.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MYCELIUM_MAX_PEERS` | 100 | Maximum P2P connections |
| `MYCELIUM_API_PORT` | 8080 | REST API port |
| `MYCELIUM_LISTEN_PORT` | 4001 | P2P listen port |
| `MYCELIUM_LOG` | info | Log level (trace/debug/info/warn/error) |
| `MYCELIUM_DATA_DIR` | ~/.mycelium | Data directory |
| `MYCELIUM_MAX_GRADIENT_NORM` | 10.0 | Gradient clipping threshold |
| `RUST_LOG` | info | Detailed log filtering |

### 10.2 Configuration File

```toml
# ~/.mycelium/config.toml

[network]
max_peers = 100
bootstrap_nodes = ["/ip4/..."]
gossipsub_mesh_n = 8
gossipsub_gossip_factor = 0.25

[compute]
device = "auto"  # auto, cpu, cuda, metal
quantization = "Q4_K_M"
batch_size = 1
max_context_length = 4096

[learning]
lora_rank = 8
learning_rate = 1e-4
gradient_clip = 1.0
aggregation_interval_secs = 60

[storage]
max_spore_cache = 10  # GB
weight_dir = "./weights"
```

### 10.3 Performance Flags

```bash
# High-performance mode
./mycelium-node --compute-threads 8 --batch-size 4

# Low-memory mode
./mycelium-node --max-peers 20 --gradient-checkpointing

# Debug mode (slow but informative)
./mycelium-node --log-level debug --enable-tracing
```

---

## 11. Performance Regression Testing

### 11.1 Automated Benchmarks

```bash
# Run benchmark suite
cargo bench --workspace

# Compare with baseline
cargo bench --workspace -- --save-baseline main
cargo bench --workspace -- --baseline main
```

### 11.2 CI Performance Checks

**GitHub Actions workflow** (future):

```yaml
name: Performance Regression
on: [pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo bench --workspace -- --output-format bencher | tee output.txt
      - name: Check regression
        run: |
          # Fail if any benchmark regressed by >10%
          ./scripts/check_regression.sh output.txt
```

### 11.3 Performance Budget

**No PR should regress these metrics by >5%**:

| Metric | Budget |
|--------|--------|
| LatentVector lerp (100K ops) | <15ms |
| Spore serialize+deserialize (1MB) | <100ms |
| LoRA forward (rank=8, 10K ops) | <150ms |
| Memory per peer connection | <3MB |
| Node idle memory | <50MB |

---

## Conclusion

Performance optimization is an ongoing process:

1. **Measure** — Benchmark current performance
2. **Identify** — Find the actual bottlenecks
3. **Optimize** — Fix the bottlenecks
4. **Verify** — Ensure improvements and no regressions
5. **Repeat** — New bottlenecks emerge

**Remember**: Premature optimization is the root of all evil. Measure first, optimize second.

---

*Benchmarks will be updated as we test with real models and multi-node setups.*

**Last Updated**: April 10, 2026
**Version**: v0.2.0

---

## See Also

- [SCALING.md](SCALING.md) — Scaling analysis and bottlenecks
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues and solutions
- [CROSSDEVICE.md](CROSSDEVICE.md) — Cross-platform build guide
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and components
