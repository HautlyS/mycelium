# CROSSDEVICE — Mycelium Cross-Platform Replication Guide

> **Purpose**: This document describes exactly how to replicate the Mycelium decentralized AI network on any device — Android, iOS, Web, CUDA GPU, AMD ROCm GPU, Apple Silicon, x86_64 Linux/Windows, and embedded ARM. Every build target, dependency, feature flag, platform constraint, and replication step is documented herein.

---

## Table of Contents

1. [Architecture Summary](#1-architecture-summary)
2. [Supported Platforms Matrix](#2-supported-platforms-matrix)
3. [Crate Dependency Graph](#3-crate-dependency-graph)
4. [Compute Backend Abstraction](#4-compute-backend-abstraction)
5. [Platform-Specific Build Instructions](#5-platform-specific-build-instructions)
6. [Web / WASM Target](#6-web--wasm-target)
7. [Android Target](#7-android-target)
8. [iOS Target](#8-ios-target)
9. [CUDA GPU Target](#9-cuda-gpu-target)
10. [AMD ROCm GPU Target](#10-amd-rocm-target)
11. [Apple Silicon (Metal) Target](#11-apple-silicon-metal-target)
12. [Docker / Container Target](#12-docker--container-target)
13. [Spore Self-Replication Protocol](#13-spore-self-replication-protocol)
14. [P2P Network Cross-Platform Considerations](#14-p2p-network-cross-platform-considerations)
15. [WebGPU Shaders (WGSL)](#15-webgpu-shaders-wgsl)
16. [Feature Flags & Conditional Compilation](#16-feature-flags--conditional-compilation)
17. [Known Limitations & Gaps](#17-known-limitations--gaps)
18. [CI/CD Pipeline](#18-cicd-pipeline)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. Architecture Summary

Mycelium is a **Rust workspace** consisting of 9 crates, built on the following technology stack:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Rust 2024 Edition | Memory-safe, cross-compilable systems language |
| **Inference Engine** | `candle` (candle-core, candle-nn, candle-transformers v0.9) | GGUF model loading, quantized matmul, transformer layers |
| **GPU Compute** | `wgpu` v24 | WebGPU compute shaders for latent operations (lerp, SiLU, RMSNorm, matmul) |
| **P2P Networking** | `libp2p` v0.54 | Kademlia DHT, gossipsub, Noise encryption, QUIC/TCP transports |
| **API Server** | `axum` v0.8 + `tokio-tungstenite` v0.26 | REST + WebSocket endpoints |
| **Serialization** | `serde` + `serde_json` + `bincode`-compatible binary format | Cross-platform data exchange |
| **Compression** | `zstd` v0.13 | Spore genome compression |
| **WASM Bridge** | `wasm-bindgen` v0.2 + `wasm-pack` | Browser integration |
| **Async Runtime** | `tokio` v1 (full) | Async I/O, channels, timers |
| **Logging** | `tracing` v0.1 + `tracing-subscriber` | Structured logging, WASM console output |

### Target Model

- **MiniMax M2.5** — 230B parameter MoE (Mixture of Experts) Transformer
- 64 layers, 64 experts (4 active per token), hidden_dim=6144
- GGUF quantized format (Q4_K_M typical)
- Context: up to 1M tokens

---

## 2. Supported Platforms Matrix

| Platform | Architecture | OS | Compute Backend | Status | Build Command |
|----------|-------------|-----|-----------------|--------|---------------|
| **Linux x86_64** | x86_64 | Linux 5.0+ | CPU (default) | ✅ Working | `cargo build --release` |
| **Linux x86_64 + CUDA** | x86_64 | Linux 5.0+ | CUDA (candle-cuda) | ⚠️ Untested | `cargo build --release --features cuda` |
| **Linux ARM64** | aarch64 | Linux 5.0+ | CPU | ✅ Works | `cargo build --release --target aarch64-unknown-linux-gnu` |
| **macOS x86_64** | x86_64 | macOS 12+ | CPU / Metal | ✅ Works | `cargo build --release --features metal` |
| **macOS Apple Silicon** | aarch64 | macOS 12+ | Metal (unified memory) | ✅ Works | `cargo build --release --features metal` |
| **Windows x86_64** | x86_64 | Windows 10+ | CPU | ⚠️ Untested | `cargo build --release` |
| **Web (Browser)** | wasm32 | Chrome 113+, Firefox 120+, Safari 16.4+ | WebGPU | ✅ Working | `wasm-pack build --target web crates/mycelium-web` |
| **Android** | aarch64 | Android 10+ (API 29+) | CPU / Vulkan (future) | 🔧 Requires setup | See §7 |
| **iOS** | aarch64 | iOS 16+ | Metal | 🔧 Requires setup | See §8 |
| **AMD ROCm** | x86_64 | Linux 5.0+ | ROCm (future) | ❌ Not implemented | See §10 |
| **Docker** | x86_64/aarch64 | Any (container) | CPU | ✅ Working | `docker build -t mycelium .` |

### Capability Tiers by Platform

| Tier | Platform | Max VRAM | Max Model Size | Role in Network |
|------|----------|----------|----------------|-----------------|
| **Tier 1 — Heavy** | Multi-GPU Server (CUDA) | 80GB+ × N | Full 230B MoE (Q4) | Layer host, expert node, coordinator |
| **Tier 2 — Medium** | Desktop GPU (CUDA/Metal) | 8–24GB | Partial layers (10–30 layers) | Pipeline parallel participant |
| **Tier 3 — Light** | Laptop CPU / Phone | 0 VRAM, 4–16GB RAM | 1–5 layers, small shards | Inference relay, spore carrier |
| **Tier 4 — Ultra-Light** | Browser (WASM/WebGPU) | 2–4GB browser limit | Latent ops only, no full model | Lightweight inference, spore germination trigger |

---

## 3. Crate Dependency Graph

```
mycelium-node (binary entry point)
├── mycelium-core          ← Foundation types (no platform-specific deps)
│   ├── serde, serde_json, uuid, sha2, bytes, chrono
│   └── [feature: cuda]    ← compile-time marker only
│
├── mycelium-substrate     ← Weight storage, GGUF parsing
│   ├── candle-core, reqwest, memmap2
│   └── Platform note: memmap2 requires mmap support (not available in WASM)
│
├── mycelium-compute       ← Inference engine, distributed router
│   ├── candle-core, candle-nn, candle-transformers
│   ├── wgpu (native + WASM)
│   ├── tokenizers (HuggingFace tokenizers)
│   ├── [feature: cuda]    → candle-core/cuda
│   ├── [feature: metal]   → candle-core/metal
│   └── [feature: wasm]    → wgpu/webgl
│
├── mycelium-hyphae        ← P2P networking
│   ├── libp2p (tokio, noise, tcp, quic, gossipsub, kad)
│   ├── mycelium-spore, mycelium-compute
│   └── Platform note: libp2p requires tokio runtime (not WASM-compatible)
│
├── mycelium-nucleus       ← Federated LoRA self-tuning
│   ├── candle-core, candle-nn, rand
│   └── Platform note: Pure compute, no platform-specific deps
│
├── mycelium-spore         ← Self-replication protocol
│   ├── zstd, crc32fast, rand, sha2
│   └── Platform note: zstd compiles to WASM; works everywhere
│
├── mycelium-fruit         ← HTTP/WebSocket API
│   ├── axum, tokio-tungstenite, tower-http
│   └── Platform note: axum requires tokio (not WASM-compatible)
│
└── mycelium-web           ← WASM browser target (cdylib)
    ├── wasm-bindgen, web-sys, js-sys, gloo-timers
    ├── wgpu (WebGPU), bytemuck
    ├── tracing-wasm, console_error_panic_hook
    ├── serde-wasm-bindgen, getrandom/js, uuid/js
    └── mycelium-core only (NO tokio, NO libp2p, NO axum)
```

### Critical Dependency Constraints

| Crate | WASM Compatible? | Reason |
|-------|-----------------|--------|
| `mycelium-core` | ✅ Yes | Pure serde types |
| `mycelium-spore` | ✅ Yes | zstd + sha2 compile to WASM |
| `mycelium-nucleus` | ⚠️ Partial | candle-core may not compile to WASM |
| `mycelium-compute` | ⚠️ Partial | wgpu works via WebGPU; candle does not |
| `mycelium-substrate` | ❌ No | `memmap2` not available in WASM |
| `mycelium-hyphae` | ❌ No | `libp2p` requires OS sockets, tokio |
| `mycelium-fruit` | ❌ No | `axum` requires tokio HTTP server |
| `mycelium-node` | ❌ No | Full binary with tokio runtime |

---

## 4. Compute Backend Abstraction

Mycelium uses a **layered compute abstraction** that allows different backends per platform:

### 4.1 Native Nodes (Linux/macOS/Windows)

```
InferenceEngine (candle)
├── GGUFLoader → candle's gguf_file parser
├── ModelWeights → quantized transformer layers
│   ├── QMatMul → candle quantized matmul
│   ├── RmsNorm → manual RMS norm implementation
│   ├── MlpOrMoe → dense or MoE FFN
│   └── LayerWeights → full transformer layer with KV cache
├── Device detection: detect_device()
│   ├── [feature: cuda] → Device::new_cuda(0)
│   ├── [feature: metal] → Device::new_metal(0)
│   └── fallback → Device::Cpu
└── DistributedTensorRouter → cross-node tensor coordination
```

### 4.2 Browser Nodes (WASM)

```
MyceliumWeb (wasm-bindgen exports)
├── GpuContext → wgpu::Instance → WebGPU adapter
├── GPU Operations (WGSL shaders):
│   ├── latent_lerp() → latent_ops.wgsl (op=0)
│   ├── matmul() → matmul.wgsl
│   ├── silu_activation() → latent_ops.wgsl (op=4)
│   └── rms_norm() → latent_ops.wgsl (op=5)
└── CPU Fallback (pure Rust):
    ├── lerp: a * (1-t) + b * t
    ├── matmul: O(m×k×n) triple loop
    ├── silu: x / (1 + exp(-x))
    └── rms_norm: x / (rms + eps)
```

### 4.3 Device Detection Code Path

From `mycelium-compute/src/lib.rs:840-869`:

```rust
pub fn detect_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }
    Ok(Device::Cpu)
}
```

### 4.4 GPU Type Classification

From `mycelium-core/src/lib.rs:227-237`:

```rust
pub enum GpuType {
    Cuda { name: String, sm_version: u32 },   // NVIDIA
    Metal { name: String },                    // Apple Silicon
    WebGPU { adapter_name: String },           // Browser
    CpuOnly,                                   // Fallback
}
```

---

## 5. Platform-Specific Build Instructions

### 5.1 Linux x86_64 (CPU)

```bash
# Prerequisites
sudo apt install build-essential pkg-config libssl-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Build
cargo build --release --workspace

# Run
./target/release/mycelium --listen /ip4/0.0.0.0/tcp/4001 --api-port 8080
```

### 5.2 Linux x86_64 + CUDA

```bash
# Prerequisites
sudo apt install build-essential pkg-config libssl-dev cuda-toolkit-12-0
rustup default stable

# Verify CUDA
nvcc --version  # Should show 12.0+
nvidia-smi      # Should show GPU

# Build with CUDA feature
cargo build --release --features cuda

# Run
./target/release/mycelium --model-path /path/to/model.gguf
```

**CUDA Requirements:**
- NVIDIA GPU with compute capability 5.0+
- CUDA Toolkit 12.0+
- `candle-core` crate compiled with `cuda` feature
- `libcudart.so` and `libcublas.so` in library path

### 5.3 macOS Apple Silicon (Metal)

```bash
# Prerequisites
xcode-select --install
brew install pkg-config openssl
rustup default stable
rustup target add aarch64-apple-darwin

# Build with Metal feature
cargo build --release --features metal

# Run (unified memory acts as VRAM)
./target/release/mycelium --model-path /path/to/model.gguf
```

**Metal Requirements:**
- macOS 12.0+ (Monterey)
- Apple Silicon (M1/M2/M3/M4)
- `candle-core` crate compiled with `metal` feature
- Unified memory: RAM = VRAM (up to 192GB on Mac Pro)

### 5.4 macOS x86_64 (Intel)

```bash
cargo build --release
# No Metal support for Intel Macs in candle; falls back to CPU
```

### 5.5 Windows x86_64

```powershell
# Prerequisites
# Install Rust from https://rustup.rs
# Install Visual Studio Build Tools with C++ workload
# Install OpenSSL for Windows

# Build
cargo build --release --workspace

# Run
.\target\release\mycelium-node.exe --listen /ip4/0.0.0.0/tcp/4001
```

**Windows Notes:**
- libp2p TCP transport works natively
- QUIC transport may require additional configuration
- CUDA support requires CUDA Toolkit for Windows
- No Metal support (Apple only)

### 5.6 Linux ARM64 (Raspberry Pi, AWS Graviton)

```bash
# On target device
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu

# Or cross-compile from x86_64
sudo apt install gcc-aarch64-linux-gnu
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
cargo build --release --target aarch64-unknown-linux-gnu
```

---

## 6. Web / WASM Target

### 6.1 Build Process

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM module
wasm-pack build --target web crates/mycelium-web --out-dir pkg

# Output files:
# pkg/mycelium_web.wasm      — WebAssembly binary
# pkg/mycelium_web.js        — JavaScript bindings
# pkg/mycelium_web.d.ts      — TypeScript definitions
# pkg/package.json            — NPM package metadata
```

### 6.2 WASM-Specific Dependencies

From `mycelium-web/Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = [
    "console", "Window", "Document", "HtmlElement", "WorkerGlobalScope"
]}
gloo-timers = { version = "0.3", features = ["futures"] }
js-sys = "0.3"
serde-wasm-bindgen = "0.6"
getrandom = { version = "0.2", features = ["js"] }
uuid = { version = "1", features = ["v4", "js"] }
console_error_panic_hook = "0.1"
tracing-wasm = "0.2"
```

### 6.3 Browser Requirements

| Browser | Minimum Version | WebGPU Support |
|---------|----------------|----------------|
| Chrome | 113+ | ✅ Native |
| Firefox | 120+ | ⚠️ Behind flag (`dom.webgpu.enabled`) |
| Safari | 16.4+ | ⚠️ Technology Preview |
| Edge | 113+ | ✅ Native (Chromium-based) |

### 6.4 JavaScript Integration

```javascript
import init, { MyceliumWeb } from './pkg/mycelium_web.js';

await init();
const node = new MyceliumWeb();
const gpuAvailable = await node.init_gpu();

if (gpuAvailable) {
    console.log('WebGPU active');
} else {
    console.log('Using CPU fallback');
}

// Latent interpolation on GPU
const result = await node.latent_lerp(vecA, vecB, 0.5);

// Matrix multiplication on GPU
const product = await node.matmul(matrixA, matrixB, m, k, n);

// SiLU activation on GPU
const activated = await node.silu_activation(input);

// RMS normalization on GPU
const normalized = await node.rms_norm(input, 1e-5);
```

### 6.5 WASM Memory Limits

| Environment | Max Memory |
|-------------|-----------|
| Chrome Desktop | 4GB (8GB with flags) |
| Chrome Android | 2GB |
| Firefox Desktop | 4GB |
| Safari Desktop | 4GB |
| Safari iOS | 1GB |

**Implication**: Full model weights (230B Q4 ≈ 114GB) cannot fit in WASM memory. Browser nodes are limited to:
- Latent-space operations (6144-dim vectors = 24KB each)
- LoRA adapter computation (rank × hidden_dim matrices)
- Spore germination triggers (metadata only)
- P2P coordination (future: WebRTC)

### 6.6 WASM Build Exclusions

The following crates are **NOT compiled to WASM**:
- `mycelium-hyphae` (libp2p — requires OS sockets)
- `mycelium-fruit` (axum — requires tokio HTTP server)
- `mycelium-substrate` (memmap2 — requires mmap syscall)
- `mycelium-node` (binary entry point — requires tokio runtime)

**Workaround for browser P2P**: Future implementation would use WebRTC via `webrtc-rs` or `libdatachannel` bindings instead of libp2p.

---

## 7. Android Target

### 7.1 Architecture Options

| Architecture | Target Triple | Status |
|-------------|--------------|--------|
| ARM64 (modern phones) | `aarch64-linux-android` | 🔧 Buildable |
| ARMv7 (older phones) | `armv7-linux-androideabi` | 🔧 Buildable |
| x86_64 (emulator) | `x86_64-linux-android` | 🔧 Buildable |

### 7.2 Prerequisites

```bash
# Install Android NDK
# Option 1: Android Studio → SDK Manager → NDK
# Option 2: Command line
sdkmanager "ndk;26.1.10909125"

# Set environment variables
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
export ANDROID_HOME=$HOME/Android/Sdk

# Install Rust Android targets
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android

# Install cargo-ndk (simplifies cross-compilation)
cargo install cargo-ndk
```

### 7.3 Build Command

```bash
# Build for ARM64
cargo ndk -t arm64-v8a -o jniLibs build --release -p mycelium-core -p mycelium-spore -p mycelium-nucleus

# Or manual cross-compile
export AR_aarch64_linux_android=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar
export CC_aarch64_linux_android=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang

cargo build --release --target aarch64-linux-android \
    -p mycelium-core \
    -p mycelium-spore \
    -p mycelium-nucleus
```

### 7.4 Android-Specific Considerations

| Component | Android Status | Notes |
|-----------|---------------|-------|
| `mycelium-core` | ✅ Works | Pure Rust, no platform deps |
| `mycelium-spore` | ✅ Works | zstd compiles to Android |
| `mycelium-nucleus` | ✅ Works | Pure compute |
| `mycelium-compute` | ⚠️ Partial | candle works on Android CPU; no Vulkan backend yet |
| `mycelium-substrate` | ⚠️ Partial | `memmap2` works on Android API 29+ |
| `mycelium-hyphae` | ⚠️ Partial | libp2p works but needs network permissions |
| `mycelium-fruit` | ⚠️ Partial | axum works but needs Android network permission |
| `mycelium-node` | ⚠️ Partial | Binary works; needs Android foreground service |
| `mycelium-web` | N/A | WASM target; use in Android Chrome via WebView |

### 7.5 AndroidManifest.xml Permissions

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.ACCESS_WIFI_STATE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

### 7.6 Android Compute Backend

For Android, the compute stack is:

```
Android App (Kotlin/Java)
├── JNI → Rust library (mycelium-core, mycelium-compute)
│   ├── candle (CPU mode)
│   └── No GPU acceleration (no CUDA/Metal on Android)
└── Future: Vulkan compute via ash crate
```

**GPU Acceleration on Android**: Not currently implemented. Options for future:
1. **Vulkan** via `ash` crate — Android 7.0+ supports Vulkan
2. **NNAPI** via `ndk` crate — Android 8.1+ neural network API
3. **GPU delegate** via TensorFlow Lite bindings

---

## 8. iOS Target

### 8.1 Architecture Options

| Architecture | Target Triple | Status |
|-------------|--------------|--------|
| ARM64 (iPhone/iPad) | `aarch64-apple-ios` | 🔧 Buildable |
| ARM64 Simulator | `aarch64-apple-ios-sim` | 🔧 Buildable |
| x86_64 Simulator (Intel Mac) | `x86_64-apple-ios` | 🔧 Buildable |

### 8.2 Prerequisites

```bash
# Install Xcode command line tools
xcode-select --install

# Install Rust iOS targets
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim
rustup target add x86_64-apple-ios

# Install cargo-lipo (universal iOS framework builder)
cargo install cargo-lipo
```

### 8.3 Build Command

```bash
# Build for device
cargo build --release --target aarch64-apple-ios \
    -p mycelium-core \
    -p mycelium-spore \
    -p mycelium-nucleus \
    --features metal

# Build universal framework (device + simulator)
cargo lipo --release -p mycelium-core -p mycelium-spore -p mycelium-nucleus

# Output: target/universal/release/libmycelium_core.a
```

### 8.4 iOS-Specific Considerations

| Component | iOS Status | Notes |
|-----------|-----------|-------|
| `mycelium-core` | ✅ Works | Pure Rust |
| `mycelium-spore` | ✅ Works | zstd compiles to iOS |
| `mycelium-nucleus` | ✅ Works | Pure compute |
| `mycelium-compute` | ✅ Works (Metal) | candle supports Metal on iOS |
| `mycelium-substrate` | ⚠️ Partial | `memmap2` works on iOS |
| `mycelium-hyphae` | ⚠️ Partial | libp2p works; needs background mode |
| `mycelium-fruit` | ⚠️ Partial | axum works; needs local network permission |
| `mycelium-node` | ⚠️ Partial | Binary works; needs iOS background execution |
| `mycelium-web` | N/A | Use in Safari via WebAssembly |

### 8.5 iOS Metal Compute

On iOS, Metal is the **only** GPU API available. The `candle-core` crate's `metal` feature works on iOS:

```rust
// From mycelium-compute/src/lib.rs:848-856
#[cfg(feature = "metal")]
{
    if let Ok(device) = Device::new_metal(0) {
        info!("Using Metal device");
        return Ok(device);
    }
}
```

**iOS Memory Limits:**

| Device | Max RAM | Available to App |
|--------|---------|-----------------|
| iPhone 15 (6GB) | 6GB | ~2-3GB |
| iPhone 15 Pro (8GB) | 8GB | ~4-5GB |
| iPad Pro M2 (16GB) | 16GB | ~8-10GB |
| iPad Pro M4 (16GB) | 16GB | ~8-10GB |

**Implication**: Even on high-end iOS devices, only partial model shards can be loaded. iOS nodes function best as:
- Pipeline parallel participants (5-15 layers)
- Spore carriers and propagators
- LoRA gradient contributors

### 8.6 iOS Background Execution

iOS restricts background execution. For continuous P2P participation:

1. **Background Modes** in `Info.plist`:
```xml
<key>UIBackgroundModes</key>
<array>
    <string>fetch</string>
    <string>processing</string>
</array>
```

2. **Background Tasks API**: Use `BGTaskScheduler` for periodic sync
3. **Foreground Service**: Use silent audio trick (not App Store compliant)

---

## 9. CUDA GPU Target

### 9.1 Supported GPU Architectures

| GPU Architecture | Compute Capability | Status |
|-----------------|-------------------|--------|
| Volta (V100) | 7.0 | ✅ Supported |
| Turing (RTX 20xx) | 7.5 | ✅ Supported |
| Ampere (RTX 30xx, A100) | 8.0/8.6 | ✅ Supported |
| Ada Lovelace (RTX 40xx) | 8.9 | ✅ Supported |
| Hopper (H100) | 9.0 | ⚠️ May need candle update |
| Kepler (GTX 6xx/7xx) | 3.0-3.5 | ❌ Too old |

### 9.2 Feature Flag

From `mycelium-compute/Cargo.toml`:

```toml
[features]
cuda = ["candle-core/cuda"]
```

From `mycelium-node/Cargo.toml`:

```toml
[features]
cuda = ["mycelium-compute/cuda"]
```

### 9.3 Build & Run

```bash
# Install CUDA Toolkit 12.0+
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
nvidia-smi

# Build with CUDA
cargo build --release --features cuda

# Run
./target/release/mycelium --model-path /path/to/model.gguf
```

### 9.4 CUDA-Specific Code Paths

Device detection (`mycelium-compute/src/lib.rs:842-848`):

```rust
#[cfg(feature = "cuda")]
{
    if let Ok(device) = Device::new_cuda(0) {
        info!("Using CUDA device");
        return Ok(device);
    }
}
```

Device info string: `"CUDA:0"` (hardcoded single-GPU)

### 9.5 Multi-GPU Support

Currently **NOT implemented**. The codebase assumes single-GPU:

```rust
Device::new_cuda(0)  // Always GPU 0
```

**For multi-GPU**, future work would:
1. Enumerate available GPUs
2. Assign layers to specific GPUs
3. Use NCCL or custom P2P for inter-GPU communication
4. Map each GPU to a virtual "node" in the topology

### 9.6 CUDA Memory Management

candle handles CUDA memory via cuMemAlloc/cuMemFree. Memory usage:
- Q4 quantized 230B model ≈ 114GB VRAM
- Requires multi-GPU for full model
- Partial layers: ~1.8GB per layer (Q4)

---

## 10. AMD ROCm Target

### 10.1 Current Status

**ROCm is NOT currently implemented** in the Mycelium codebase. The `candle` framework does not have native ROCm support.

### 10.2 Implementation Options

#### Option A: Wait for candle ROCm support
- Track `candle` crate ROCm feature requests
- https://github.com/huggingface/candle/issues

#### Option B: Use Vulkan as Universal GPU Backend
- `wgpu` supports Vulkan natively
- AMD GPUs have excellent Vulkan drivers on Linux
- Would require implementing compute kernels in WGSL instead of CUDA kernels

```toml
# Add to mycelium-compute/Cargo.toml
[features]
vulkan = ["wgpu/vulkan"]  # wgpu already includes Vulkan support
```

#### Option C: HIP Translation Layer
- AMD's HIP can translate CUDA code to ROCm
- Requires `hipblaslt` and `rocblas` libraries
- candle would need to compile against HIP instead of CUDA

### 10.3 Recommended Path: wgpu/Vulkan

Since Mycelium already uses `wgpu` for WebGPU compute shaders, the most consistent path for AMD ROCm support is:

1. **Extend `mycelium-web`'s wgpu compute pipeline** to native
2. **Use WGSL shaders** (`matmul.wgsl`, `latent_ops.wgsl`, `attention.wgsl`) for native GPU compute
3. **wgpu's Vulkan backend** handles AMD GPUs transparently

```rust
// Native wgpu compute (similar to WASM implementation)
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends: wgpu::Backends::VULKAN,  // Force Vulkan for AMD
    ..Default::default()
});
```

**Advantages**:
- Same code path as browser (WGSL shaders)
- No CUDA dependency
- Cross-platform (Linux, Windows, macOS via MoltenVK)

**Disadvantages**:
- WGSL shaders are currently simplified (need full matmul implementation)
- candle's optimized quantized matmul would not be used
- Performance may be lower than native CUDA kernels

### 10.4 AMD GPU Requirements

| Component | Requirement |
|-----------|------------|
| GPU | RDNA2 (RX 6000+) or RDNA3 (RX 7000+) recommended |
| Driver | Mesa Vulkan drivers (Linux) or AMD Adrenalin (Windows) |
| VRAM | 16GB+ recommended for meaningful layer participation |

---

## 11. Apple Silicon (Metal) Target

### 11.1 Supported Chips

| Chip | Cores | Unified Memory | Metal Support |
|------|-------|---------------|---------------|
| M1 | 8 | 8-16GB | ✅ |
| M1 Pro | 16 | 16-32GB | ✅ |
| M1 Max | 32 | 32-64GB | ✅ |
| M1 Ultra | 64 | 64-128GB | ✅ |
| M2 | 10 | 8-24GB | ✅ |
| M2 Pro | 19 | 16-32GB | ✅ |
| M2 Max | 38 | 32-96GB | ✅ |
| M2 Ultra | 76 | 64-192GB | ✅ |
| M3 | 10 | 8-24GB | ✅ |
| M3 Pro | 18 | 18-36GB | ✅ |
| M3 Max | 40 | 36-128GB | ✅ |
| M4 | 10 | 16-32GB | ✅ |
| M4 Pro | 16 | 24-48GB | ✅ |
| M4 Max | 40 | 36-128GB | ✅ |

### 11.2 Feature Flag

```toml
[features]
metal = ["candle-core/metal"]
```

### 11.3 Build & Run

```bash
# Build with Metal
cargo build --release --features metal

# Run
./target/release/mycelium --model-path /path/to/model.gguf
```

### 11.4 Metal-Specific Advantages

1. **Unified Memory**: No VRAM/RAM copy needed — GPU reads system memory directly
2. **Zero-Copy Tensors**: candle's Metal backend maps tensors directly to GPU buffers
3. **High Bandwidth**: M2 Ultra = 800 GB/s memory bandwidth (comparable to A100)
4. **Large Models**: M2 Ultra with 192GB can load full Q4 230B model

### 11.5 Metal Performance Expectations

| Chip | Layers/sec (Q4) | Tokens/sec |
|------|----------------|------------|
| M1 (8GB) | 5-10 | ~2-5 |
| M2 Max (96GB) | 30-40 | ~10-15 |
| M2 Ultra (192GB) | 64 (full model) | ~15-20 |
| M4 Max (128GB) | 50-60 | ~15-25 |

---

## 12. Docker / Container Target

### 12.1 Dockerfile Analysis

From the existing `Dockerfile`:

```dockerfile
# Stage 1: Builder
FROM rust:1.75-slim as builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY shaders ./shaders
RUN cargo build --release --workspace

# Stage 2: Production
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/false mycelium
COPY --from=builder /build/target/release/mycelium-node /usr/local/bin/
RUN mkdir -p /data && chown mycelium:mycelium /data
USER mycelium
EXPOSE 8080 4001
ENV MYCELIUM_DATA_DIR=/data
ENV RUST_LOG=info
VOLUME ["/data"]
ENTRYPOINT ["/usr/local/bin/mycelium-node"]
CMD ["--api-port", "8080", "--listen", "0.0.0.0:4001", "--data-dir", "/data"]
```

### 12.2 Build & Run

```bash
# Build
docker build -t mycelium .

# Run (CPU mode)
docker run -p 8080:8080 -p 4001:4001 mycelium

# Run with model (mount GGUF file)
docker run -p 8080:8080 -p 4001:4001 \
    -v /path/to/model.gguf:/model.gguf:ro \
    mycelium --model-path /model.gguf

# Run with persistent data
docker run -p 8080:8080 -p 4001:4001 \
    -v mycelium-data:/data \
    mycelium
```

### 12.3 Docker + CUDA (NVIDIA Container Toolkit Required)

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as base
FROM rust:1.75 as builder
# ... same build stage ...

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
COPY --from=builder /build/target/release/mycelium-node /usr/local/bin/
# CUDA runtime libraries included in base image
```

```bash
# Build CUDA Docker image
docker build -f Dockerfile.cuda -t mycelium-cuda .

# Run with GPU access
docker run --gpus all -p 8080:8080 -p 4001:4001 mycelium-cuda \
    --model-path /model.gguf
```

### 12.4 Docker Compose for Multi-Node

```yaml
version: "3.8"
services:
  mycelium-1:
    image: mycelium
    ports:
      - "8081:8080"
      - "4001:4001"
    volumes:
      - data1:/data
    environment:
      - RUST_LOG=info

  mycelium-2:
    image: mycelium
    ports:
      - "8082:8080"
      - "4002:4001"
    volumes:
      - data2:/data
    depends_on:
      - mycelium-1
    command: ["--bootstrap", "/ip4/mycelium-1/tcp/4001"]

volumes:
  data1:
  data2:
```

### 12.5 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mycelium
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mycelium
  template:
    metadata:
      labels:
        app: mycelium
    spec:
      containers:
      - name: mycelium
        image: mycelium:latest
        ports:
        - containerPort: 8080
        - containerPort: 4001
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: mycelium-data
```

---

## 13. Spore Self-Replication Protocol

### 13.1 Spore Structure

From `mycelium-core/src/lib.rs:1138-1188`:

```rust
pub struct Spore {
    pub id: Uuid,
    pub genome: SporeGenome,       // Compressed GGUF shard
    pub instincts: Option<LoRAAdapter>,  // Learned behaviors
    pub model_config: ModelConfig,
    pub layer_range: (usize, usize),
    pub expert_ids: Vec<usize>,
    pub created_at: DateTime<Utc>,
    pub parent: NodeId,
    pub generation: u32,
}

pub struct SporeGenome {
    pub data: Vec<u8>,              // zstd-compressed weights
    pub quant_bits: u8,
    pub hash: String,               // SHA-256
    pub decompressed_size: u64,
}
```

### 13.2 Binary Serialization Format

```
+--------+--------+--------+--------+
| 0x4D   | 0x59   | 0x43   | 0x45   |  Magic: "MYCE"
+--------+--------+--------+--------+
| Spore ID (16 bytes UUID)           |
+--------+--------+--------+--------+
| Layer Range (u64, u64)            |
+--------+--------+--------+--------+
| Expert Count (u32)                 |
| Expert IDs (u32 × count)           |
+--------+--------+--------+--------+
| Genome Length (u64)                |
| Genome Data (zstd compressed)      |
+--------+--------+--------+--------+
| Quant Bits (u8)                    |
+--------+--------+--------+--------+
| Decompressed Size (u64)            |
+--------+--------+--------+--------+
| CRC32 (u32)                        |  Footer: integrity check
+--------+--------+--------+--------+
```

### 13.3 Replication Flow

```
1. Source node detects capacity excess:
   - VRAM utilization < threshold
   - Uptime > minimum stable period
   - LoRA adapter shows improvement

2. Package current state into Spore:
   - Compress model shard with zstd
   - Attach LoRA adapter (instincts)
   - Compute SHA-256 hash
   - Add CRC32 footer

3. Broadcast availability via gossipsub:
   - Topic: "mycelium/spore"
   - Message: SporeAvailable { spore_id, model_name, shard_count, total_size_mb }

4. Target nodes receive broadcast:
   - Check available storage
   - Check compute compatibility
   - Request spore if suitable

5. Chunked transfer:
   - SporeChunk { spore_id, chunk_idx, data }
   - Verify each chunk's integrity
   - Reassemble and verify full genome hash

6. Germination:
   - Decompress genome
   - Load weights into candle
   - Apply LoRA adapter
   - Join network as full participant
```

### 13.4 Cross-Platform Spore Compatibility

| Source Platform | Target Platform | Compatible? | Notes |
|----------------|----------------|-------------|-------|
| Linux CUDA | Linux CUDA | ✅ Yes | Identical binary format |
| Linux CUDA | macOS Metal | ✅ Yes | GGUF is platform-agnostic |
| Linux CUDA | WASM | ⚠️ Partial | WASM can receive metadata, not full weights |
| macOS Metal | Linux CUDA | ✅ Yes | GGUF format is universal |
| WASM | Native | ❌ No | WASM cannot send full model weights |
| Any | Android | ✅ Yes | GGUF works on Android CPU |
| Any | iOS | ✅ Yes | GGUF works with Metal on iOS |

**Key Insight**: GGUF is a **platform-agnostic binary format**. Weights serialized on one platform can be loaded on any other platform that supports the same quantization type.

---

## 14. P2P Network Cross-Platform Considerations

### 14.1 libp2p Transport Matrix

| Transport | Linux | macOS | Windows | Android | iOS | WASM |
|-----------|-------|-------|---------|---------|-----|------|
| TCP | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| QUIC (UDP) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| WebSocket | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* |
| WebRTC | ❌** | ❌** | ❌** | ❌** | ❌** | ✅ |

- *: WASM would need WebSocket transport (not yet implemented)
- **: WebRTC support would require `webrtc-rs` crate integration

### 14.2 NAT Traversal

```rust
// From libp2p configuration in mycelium-hyphae
libp2p = { version = "0.54", features = [
    "tokio", "noise", "tcp", "quic", "yamux",
    "gossipsub", "kad", "identify", "ping", "relay",
    "autonat", "dcutr", "macros",
]}
```

| Mechanism | Purpose | Platforms |
|-----------|---------|-----------|
| AutoNAT | Detect if behind NAT | All native platforms |
| DCUtR | Hole-punching through NAT | All native platforms |
| Relay | Fallback for unreachable nodes | All platforms |
| Kademlia DHT | Peer discovery | All native platforms |

### 14.3 Mobile Network Considerations

**Android:**
- Mobile data may block P2P ports (4001)
- WiFi NAT traversal usually works
- Use WebSocket transport as fallback (port 443)

**iOS:**
- Background P2P is limited by iOS
- Cellular data may restrict UDP (QUIC)
- Use TCP/WebSocket for reliability

### 14.4 Browser P2P (Future)

For full browser-to-browser P2P:

```
WebRTC (via webrtc-rs or libdatachannel)
├── ICE candidates for NAT traversal
├── STUN servers for public IP discovery
├── TURN servers as relay fallback
└── DataChannel for libp2p-compatible streams
```

This would replace `libp2p` in the WASM target with a WebRTC-based transport.

---

## 15. WebGPU Shaders (WGSL)

### 15.1 Shader Inventory

| Shader | File | Operations | Workgroup Size |
|--------|------|-----------|----------------|
| Latent Ops | `shaders/latent_ops.wgsl` | lerp, normalize, blend, SiLU, RMSNorm | 256 × 1 × 1 |
| Matrix Multiply | `shaders/matmul.wgsl` | C = A × B | 16 × 16 × 1 |
| Attention | `shaders/attention.wgsl` | Multi-head attention | 8 × 8 × 1 |

### 15.2 Latent Ops Shader

From `shaders/latent_ops.wgsl`:

```wgsl
struct Params {
    dim: u32,
    operation: u32,  // 0=lerp, 1=normalize, 2=blend, 4=silu, 5=rms_norm
    t: f32,
    scale: f32,
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.dim) { return; }

    switch params.operation {
        case 0u: {  // Lerp
            output[idx] = input_a[idx] * (1.0 - params.t) + input_b[idx] * params.t;
        }
        case 4u: {  // SiLU
            let x = input_a[idx];
            let sig = 1.0 / (1.0 + exp(-x));
            output[idx] = x * sig;
        }
        case 5u: {  // RMSNorm
            // Simplified — production needs two-pass reduction
        }
        // ...
    }
}
```

### 15.3 Shader Compilation Targets

| Target | Backend | Shader Format |
|--------|---------|--------------|
| Native (Linux/macOS/Windows) | wgpu Vulkan/Metal/DirectX12 | WGSL (compiled at runtime) |
| Native + CUDA feature | wgpu (not candle) | WGSL |
| WASM (Browser) | wgpu WebGPU | WGSL (native browser format) |
| Android (future) | wgpu Vulkan | WGSL → SPIR-V → Vulkan |
| iOS (future) | wgpu Metal | WGSL → MSL → Metal |

**Key**: wgpu automatically translates WGSL to the native shader format:
- Vulkan → SPIR-V
- Metal → MSL (Metal Shading Language)
- DirectX 12 → DXIL
- WebGPU → WGSL (native)

### 15.4 Shader Portability Notes

1. **No push constants** — Uses uniform buffer for params (better WASM compatibility)
2. **Single-pass normalization** — Current RMSNorm is approximate; production needs two-pass reduction
3. **No shared memory** — Shaders don't use workgroup shared memory (limits optimization but maximizes portability)
4. **No atomics** — Avoids atomic operations for broader compatibility

---

## 16. Feature Flags & Conditional Compilation

### 16.1 Workspace-Level Feature Flags

From `Cargo.toml`:

```toml
[workspace.dependencies]
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
wgpu = "24"
```

### 16.2 Per-Crate Feature Flags

**mycelium-core:**
```toml
[features]
cuda = []  # Marker only — actual CUDA support in mycelium-compute
```

**mycelium-compute:**
```toml
[features]
default = []
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]
wasm = ["wgpu/webgl"]
```

**mycelium-node:**
```toml
[features]
default = []
cuda = ["mycelium-compute/cuda"]
metal = ["mycelium-compute/metal"]
```

### 16.3 cfg Attributes in Source Code

| cfg Attribute | Purpose | Location |
|--------------|---------|----------|
| `#[cfg(feature = "cuda")]` | Enable CUDA device creation | `mycelium-compute/src/lib.rs:842-848` |
| `#[cfg(feature = "metal")]` | Enable Metal device creation | `mycelium-compute/src/lib.rs:850-856` |
| `#[cfg(target_os = "linux")]` | Linux-specific system calls | `mycelium-core/src/lib.rs:85-91, 110-126` |
| `#[cfg(target_os = "macos")]` | macOS-specific system calls | `mycelium-core/src/lib.rs:92-101, 127-137, 163-187` |
| `#[cfg(not(any(target_os = "linux", target_os = "macos")))]` | Fallback for other OSes | `mycelium-core/src/lib.rs:102-106, 138-141` |

### 16.4 Platform-Specific Hardware Detection

From `mycelium-core/src/lib.rs:60-224`:

```rust
impl NodeCapabilities {
    pub fn auto_detect() -> Self {
        let ram_mb = Self::detect_ram_mb();      // OS-specific
        let (gpu_type, vram_mb, compute_units) = Self::detect_gpu();  // OS-specific
        let cpu_cores = Self::detect_cpu_cores();  // OS-specific
        // ...
    }
}
```

| Platform | RAM Detection | GPU Detection |
|----------|--------------|---------------|
| Linux | `/proc/meminfo` → `MemTotal` | CUDA check (if feature enabled) |
| macOS | `sysctl -n hw.memsize` | `system_profiler SPDisplaysDataType` |
| Other | Default 8192 MB | `GpuType::CpuOnly` |

---

## 17. Known Limitations & Gaps

### 17.1 Current Implementation Gaps

| Gap | Severity | Affected Platforms | Workaround |
|-----|----------|-------------------|------------|
| CUDA path untested | Medium | Linux + NVIDIA | Use CPU mode; fallback works |
| AMD ROCm not supported | High | Linux/Windows + AMD | Use CPU mode; Vulkan path planned |
| WASM cannot run libp2p | Medium | All browsers | Use WebSocket transport (future) |
| WASM cannot load full model | High | All browsers | Latent ops only; coordinate with native nodes |
| Single-GPU only | Medium | All CUDA/Metal systems | Run multiple processes per machine |
| No iOS background P2P | Medium | iOS | Foreground-only; periodic sync |
| Android GPU not used | Medium | Android | CPU inference; Vulkan planned |
| Windows untested | Low | Windows | Should work; needs validation |
| matmul WGSL is simplified | Medium | WebGPU | CPU fallback available |
| No Byzantine fault tolerance | High | All platforms | Gradient aggregation is naive |
| No differential privacy | High | All platforms | Raw gradients shared |

### 17.2 Model Size vs Platform Capability

| Platform | Max Layers (Q4) | Max Layers (Q8) | Full Model Possible? |
|----------|----------------|----------------|---------------------|
| M2 Ultra 192GB | 64 (full) | 64 (full) | ✅ Yes |
| A100 80GB | ~40 | ~20 | ❌ No |
| RTX 4090 24GB | ~12 | ~6 | ❌ No |
| M2 Max 96GB | ~50 | ~25 | ❌ No |
| iPhone 16GB RAM | ~8 | ~4 | ❌ No |
| Android 8GB RAM | ~4 | ~2 | ❌ No |
| Browser 4GB | 0 (latent only) | 0 | ❌ No |

**Conclusion**: Full 230B model requires either M2 Ultra (192GB unified) or multi-GPU setup. All other platforms must participate via distributed inference.

---

## 18. CI/CD Pipeline

### 18.1 GitHub Actions Configuration

From `.github/workflows/ci.yml`:

| Job | Runner | What It Builds | Output |
|-----|--------|---------------|--------|
| `build-web` | ubuntu-latest | WASM → GitHub Pages | Deployed to `gh-pages` branch |
| `build-native` | ubuntu-latest | CPU binary + tests | Artifact: `mycelium-native` |
| `build-wasm` | ubuntu-latest | WASM package | Artifact: `mycelium-web` |
| `build-cuda` | ubuntu-latest | CUDA binary | Artifact: `mycelium-cuda` |
| `release` | ubuntu-latest | Combines artifacts | GitHub Release (draft) |

### 18.2 Adding Platform-Specific CI Jobs

```yaml
# Add to .github/workflows/ci.yml

build-macos-metal:
  runs-on: macos-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: cargo build --release --features metal --workspace
    - run: cargo test --workspace --lib

build-android:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: rustup target add aarch64-linux-android
    - uses: nttld/setup-ndk@v1
      with:
        ndk-version: r26b
    - run: cargo ndk -t arm64-v8a build --release -p mycelium-core

build-ios:
  runs-on: macos-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - run: rustup target add aarch64-apple-ios
    - run: cargo build --release --target aarch64-apple-ios \
             -p mycelium-core -p mycelium-spore --features metal
```

### 18.3 Release Artifacts Per Platform

| Platform | Artifact Name | Format |
|----------|--------------|--------|
| Linux x86_64 CPU | `mycelium-linux-x86_64.tar.gz` | Binary + checksum |
| Linux x86_64 CUDA | `mycelium-linux-x86_64-cuda.tar.gz` | Binary + checksum |
| macOS x86_64 | `mycelium-macos-x86_64.tar.gz` | Binary + checksum |
| macOS Apple Silicon | `mycelium-macos-aarch64.tar.gz` | Binary + checksum |
| Windows x86_64 | `mycelium-windows-x86_64.zip` | Binary + checksum |
| WASM | `mycelium-web.tar.gz` | WASM + JS + types |
| Android ARM64 | `mycelium-android-arm64.aar` | Android Archive |
| iOS ARM64 | `mycelium-ios.xcframework` | XCFramework |
| Docker | `hautlys/mycelium:latest` | Container image |

---

## 19. Troubleshooting

### 19.1 Common Build Issues

**Problem**: `candle-core` CUDA feature fails to compile
```
error: failed to run custom build command for cudarc-sys
```
**Solution**: Ensure CUDA Toolkit 12.0+ is installed and `nvcc` is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem**: `wgpu` fails on Linux without GPU drivers
```
error: no suitable adapter found
```
**Solution**: Install Mesa Vulkan drivers:
```bash
sudo apt install mesa-vulkan-drivers vulkan-tools
```

**Problem**: WASM build includes tokio-dependent crates
```
error[E0432]: unresolved import `tokio`
```
**Solution**: Only build `mycelium-web` for WASM, not the full workspace:
```bash
wasm-pack build --target web crates/mycelium-web
# NOT: wasm-pack build --target web . (entire workspace)
```

**Problem**: `memmap2` fails on WASM
```
error: target is not supported
```
**Solution**: `mycelium-substrate` cannot be compiled to WASM. Exclude it from WASM builds.

### 19.2 Runtime Issues

**Problem**: WebGPU not available in browser
```
WebGPU not available, falling back to CPU
```
**Solution**: 
- Chrome 113+: WebGPU enabled by default
- Firefox: Set `dom.webgpu.enabled` to `true` in `about:config`
- Safari: Use Technology Preview

**Problem**: libp2p nodes can't connect
```
Peer joined but no connection established
```
**Solution**: 
- Ensure port 4001 (TCP) and 4001 (UDP for QUIC) are open
- Check firewall rules
- Use `--bootstrap` to connect to known peers

**Problem**: Model too large for VRAM
```
CUDA out of memory
```
**Solution**:
- Use lower quantization (Q4 instead of Q8)
- Run in distributed mode across multiple nodes
- Use CPU fallback (slower but works)

### 19.3 Platform-Specific Issues

**Android:**
- Ensure `INTERNET` permission is granted
- Use `cargo-ndk` for simplified cross-compilation
- Test on physical device (emulator may have GPU issues)

**iOS:**
- Metal requires `MTLDevice` entitlement
- Background P2P requires `UIBackgroundModes`
- Test on physical device (simulator uses different GPU path)

**AMD ROCm:**
- Currently unsupported — use CPU mode
- Track `candle` ROCm feature development

**Windows:**
- OpenSSL may require manual installation via vcpkg
- libp2p QUIC may need Windows Firewall exception

---

## Appendix A: Complete Build Matrix

| Platform | Arch | OS | Command | Features | Binary Size (release) |
|----------|------|-----|---------|----------|---------------------|
| Linux CPU | x86_64 | Ubuntu 22.04+ | `cargo build --release` | — | ~15MB (stripped) |
| Linux CUDA | x86_64 | Ubuntu 22.04+ | `cargo build --release --features cuda` | cuda | ~20MB |
| Linux ARM64 | aarch64 | Ubuntu 22.04+ | `cargo build --release --target aarch64-unknown-linux-gnu` | — | ~15MB |
| macOS Intel | x86_64 | macOS 12+ | `cargo build --release` | — | ~18MB |
| macOS Metal | aarch64 | macOS 12+ | `cargo build --release --features metal` | metal | ~18MB |
| Windows CPU | x86_64 | Windows 10+ | `cargo build --release` | — | ~20MB |
| WASM | wasm32 | Browser | `wasm-pack build --target web crates/mycelium-web` | — | ~2MB |
| Android ARM64 | aarch64 | Android 10+ | `cargo ndk -t arm64-v8a build --release` | — | ~12MB |
| iOS ARM64 | aarch64 | iOS 16+ | `cargo build --release --target aarch64-apple-ios` | metal | ~15MB |
| Docker | x86_64 | Any | `docker build -t mycelium .` | — | ~80MB (image) |

## Appendix B: Environment Variables

| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `RUST_LOG` | Log level | `info` | No |
| `MYCELIUM_DATA_DIR` | Substrate storage path | `~/.mycelium/data` | No |
| `CUDA_VISIBLE_DEVICES` | Select GPU | `0` | No (CUDA only) |
| `WGPU_BACKEND` | Force wgpu backend | Auto-detect | No |
| `ANDROID_NDK_HOME` | Android NDK path | — | Yes (Android builds) |
| `CC_aarch64_linux_android` | Cross-compile C compiler | — | Yes (Android manual) |

## Appendix C: Network Ports

| Port | Protocol | Purpose | Configurable |
|------|----------|---------|-------------|
| 4001 | TCP/QUIC | P2P libp2p listening | `--listen` |
| 8080 | HTTP/WS | REST API + WebSocket streaming | `--api-port` |

## Appendix D: File Formats

| Format | Extension | Purpose | Cross-Platform |
|--------|-----------|---------|---------------|
| GGUF | `.gguf` | Model weights | ✅ Universal |
| Tokenizer | `.json` | HuggingFace tokenizer | ✅ Universal |
| Spore | `.spore` (custom) | Replication package | ✅ Universal |
| LoRA | `.json` + `.bin` | Adapter weights | ✅ Universal |
| Substrate | `.dat` | Persistent storage | ✅ Universal |

---

*Document generated from Mycelium v0.2.0 source analysis — 2026-04-10*

*ॐ तारे तुत्तारे तुरे स्वा*
