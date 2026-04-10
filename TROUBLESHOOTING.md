# MYCELIUM — Troubleshooting Guide

> *"Every error is a learning opportunity."*

This document covers common issues, error messages, and their solutions for Mycelium nodes.

---

## Table of Contents

1. [Quick Diagnostics](#1-quick-diagnostics)
2. [Build Issues](#2-build-issues)
3. [Runtime Issues](#3-runtime-issues)
4. [Network Issues](#4-network-issues)
5. [Model Loading Issues](#5-model-loading-issues)
6. [Inference Issues](#6-inference-issues)
7. [Memory Issues](#7-memory-issues)
8. [GPU Issues](#8-gpu-issues)
9. [WASM/Browser Issues](#9-wasmbrowser-issues)
10. [Spore Protocol Issues](#10-spore-protocol-issues)
11. [API Issues](#11-api-issues)
12. [Common Error Messages](#12-common-error-messages)
13. [Getting Help](#13-getting-help)

---

## 1. Quick Diagnostics

### 1.1 Health Check

```bash
# Check if node is running
curl http://localhost:8080/health

# Expected response:
# {"status": "ok", "uptime_seconds": 123}

# Check node status
curl http://localhost:8080/status

# Check peer connections
curl http://localhost:8080/peers
```

### 1.2 Log Analysis

```bash
# Run with detailed logging
RUST_LOG=debug ./target/release/mycelium-node

# Filter for errors only
RUST_LOG=mycelium_node=error ./target/release/mycelium-node

# Search logs for specific errors
grep -i "error" ~/.mycelium/logs/*.log
```

### 1.3 Resource Check

```bash
# Check memory usage
ps aux | grep mycelium

# Check network connections
netstat -an | grep 4001

# Check GPU (NVIDIA)
nvidia-smi
```

---

## 2. Build Issues

### 2.1 Rust Version Too Old

**Error**: `error[E0xxx]: feature has been stabilized in version 1.xx`

**Solution**:
```bash
rustup update stable
rustup default stable
rustc --version  # Should be 1.75+
```

### 2.2 Missing System Dependencies

**Error**: `linker cc not found` or `openssl-sys build failed`

**Solution** (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install build-essential pkg-config libssl-dev
```

**Solution** (macOS):
```bash
xcode-select --install
brew install pkg-config openssl
```

**Solution** (Windows):
- Install Visual Studio Build Tools with C++ workload
- Install OpenSSL via vcpkg

### 2.3 CUDA Build Failures

**Error**: `cuda runtime error` or `candle-cuda build failed`

**Checklist**:
1. CUDA Toolkit installed? `nvcc --version`
2. GPU available? `nvidia-smi`
3. Correct feature flag? `cargo build --features cuda`
4. CUDA in PATH? `echo $PATH | grep cuda`

**Solution**:
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild
cargo clean
cargo build --release --features cuda
```

### 2.4 WASM Build Failures

**Error**: `wasm-pack command not found` or build errors

**Solution**:
```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM
wasm-pack build --target web crates/mycelium-web

# If compilation fails, check browser compatibility
# WebGPU requires Chrome 113+, Firefox 120+, Safari 16.4+
```

### 2.5 Dependency Resolution Failures

**Error**: `failed to select a version for xxx`

**Solution**:
```bash
# Clear cargo cache
cargo clean
rm -rf ~/.cargo/registry/cache

# Update lock file
cargo update

# Rebuild
cargo build --release
```

---

## 3. Runtime Issues

### 3.1 Node Won't Start

**Error**: `Failed to bind to address` or `Address already in use`

**Cause**: Port already in use

**Solution**:
```bash
# Check what's using the port
lsof -i :4001  # P2P port
lsof -i :8080  # API port

# Kill conflicting process or use different ports
./target/release/mycelium-node --api-port 8081 --listen 0.0.0.0:4002
```

### 3.2 Node Crashes on Startup

**Error**: `panic at ...` or `segmentation fault`

**Debug steps**:
```bash
# Run with backtrace
RUST_BACKTRACE=1 ./target/release/mycelium-node

# Run with full backtrace
RUST_BACKTRACE=full ./target/release/mycelium-node

# Run with debug logging
RUST_LOG=debug ./target/release/mycelium-node
```

**Common causes**:
- Invalid configuration file
- Corrupted data directory
- Incompatible model file

**Solution**:
```bash
# Reset data directory (WARNING: deletes local data)
rm -rf ~/.mycelium
./target/release/mycelium-node
```

### 3.3 High CPU Usage

**Symptom**: Node consuming 100% CPU

**Possible causes**:
1. **Too many peers** — Limit connections
2. **Active inference** — Normal during generation
3. **Infinite loop** — Bug (report it)

**Solution**:
```bash
# Limit peer count
./target/release/mycelium-node --max-peers 50

# Check what the node is doing
RUST_LOG=mycelium_node=info ./target/release/mycelium-node
```

### 3.4 Node Not Responding to API

**Symptom**: `curl: (7) Failed to connect to localhost port 8080`

**Check**:
```bash
# Is the process running?
ps aux | grep mycelium

# Is the port listening?
netstat -an | grep 8080

# Check firewall
sudo ufw status
```

**Solution**:
```bash
# Restart node
pkill mycelium-node
./target/release/mycelium-node --api-port 8080

# Check firewall rules
sudo ufw allow 8080/tcp
```

---

## 4. Network Issues

### 4.1 Can't Connect to Peers

**Error**: `No peers connected` or `Connection refused`

**Checklist**:
1. Firewall open on port 4001?
2. NAT configured (port forwarding)?
3. Bootstrap nodes reachable?
4. Internet connection working?

**Solution**:
```bash
# Test connectivity
curl -v /ip4/YOUR_PUBLIC_IP/tcp/4001

# Check NAT type
# If behind NAT, ensure UPnP or port forwarding is enabled

# Manually specify bootstrap nodes
./target/release/mycelium-node \
    --bootstrap /ip4/BOOTSTRAP_IP/tcp/4001/p2p/BOOTSTRAP_PEER_ID
```

### 4.2 Few Peer Connections

**Symptom**: Only 1-2 peers when expecting more

**Causes**:
- New node (takes time to discover peers)
- Behind strict NAT
- Bootstrap nodes unreachable

**Solution**:
```bash
# Wait 5-10 minutes for DHT discovery
# Manually add known peers
curl -X POST http://localhost:8080/peers/add \
    -d '{"multiaddr": "/ip4/PEER_IP/tcp/4001/p2p/PEER_ID"}'

# Check NAT traversal status
curl http://localhost:8080/status | grep -i nat
```

### 4.3 High Latency to Peers

**Symptom**: Messages taking >500ms

**Causes**:
- Geographically distant peers
- Network congestion
- Slow peers

**Solution**:
```bash
# Check peer latencies
curl http://localhost:8080/peers | jq '.[] | {id, latency}'

# Prefer regional peers
# Configure region-specific bootstrap nodes
./target/release/mycelium-node \
    --bootstrap /ip4/REGIONAL_BOOTSTRAP/tcp/4001/p2p/ID
```

### 4.4 DHT Lookup Failures

**Error**: `Key not found in DHT`

**Causes**:
- Key was never published
- Node is new (DHT not populated)
- Network partition

**Solution**:
```bash
# Wait for DHT to populate (5-10 minutes)
# Check DHT statistics
curl http://localhost:8080/status | jq '.dht'

# Ensure you're connected to the right network
# Check network ID/protocol version
```

---

## 5. Model Loading Issues

### 5.1 Model File Not Found

**Error**: `Failed to load model: No such file or directory`

**Solution**:
```bash
# Verify file exists
ls -lh /path/to/model.gguf

# Check permissions
chmod 644 /path/to/model.gguf

# Use absolute path
./target/release/mycelium-node --model /absolute/path/to/model.gguf
```

### 5.2 Model File Corrupted

**Error**: `Invalid GGUF magic` or `GGUF parse error`

**Solution**:
```bash
# Verify file integrity
sha256sum /path/to/model.gguf

# Compare with expected hash (from source)
# Re-download if hash doesn't match

# Check file size (should match expected)
ls -lh /path/to/model.gguf
```

### 5.3 Insufficient Memory for Model

**Error**: `Out of memory` or `Failed to allocate`

**Solution**:
```bash
# Check available memory
free -h

# Use smaller quantization
# Q4_K_M: ~115GB for 230B model
# Q2_K: ~58GB for 230B model

# Use swap space (slow but works)
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Or run without model (P2P only)
./target/release/mycelium-node  # No --model flag
```

### 5.4 Weight Verification Failed

**Error**: `Weight verification failed: hash mismatch`

**Cause**: Model hash doesn't match registry entry

**Solution**:
```bash
# Check weight registry
cat ~/.mycelium/weight_registry.toml

# Update with correct hash
sha256sum /path/to/model.gguf

# Add to registry
echo '
[[weights]]
name = "minimax-m2.5-q4"
sha256 = "ACTUAL_HASH_HERE"
' >> ~/.mycelium/weight_registry.toml
```

---

## 6. Inference Issues

### 6.1 Generation Returns Empty Response

**Symptom**: `curl -X POST http://localhost:8080/generate` returns empty text

**Possible causes**:
- Model not loaded
- Tokenizer issue
- Generation parameters too restrictive

**Solution**:
```bash
# Check if model is loaded
curl http://localhost:8080/status | jq '.model'

# Check generation parameters
curl -X POST http://localhost:8080/generate \
    -d '{"prompt": "Hello", "max_tokens": 32, "temperature": 0.7}'

# Check logs for errors
tail -f ~/.mycelium/logs/*.log | grep -i inference
```

### 6.2 Very Slow Generation

**Symptom**: <0.1 tokens/second

**Causes**:
- Running on CPU with large model
- Too many concurrent requests
- Memory swapping

**Solution**:
```bash
# Check device being used
curl http://localhost:8080/status | jq '.device'

# If should be using GPU but isn't:
# Rebuild with GPU feature
cargo build --release --features cuda  # or metal

# Reduce concurrent requests
# Limit batch size in config
```

### 6.3 Gibberish Output

**Symptom**: Model produces nonsensical text

**Causes**:
- Wrong tokenizer
- Corrupted model weights
- Temperature too high

**Solution**:
```bash
# Verify tokenizer
ls -lh /path/to/tokenizer.json

# Lower temperature
curl -X POST http://localhost:8080/generate \
    -d '{"prompt": "Hello", "temperature": 0.1}'

# Verify model integrity
# Re-download if corrupted
```

### 6.4 Context Window Exceeded

**Error**: `Context length exceeded` or `KV cache full`

**Solution**:
```bash
# Increase context window (requires more memory)
./target/release/mycelium-node \
    --max-context-length 8192

# Or use shorter inputs
# Or enable context truncation
curl -X POST http://localhost:8080/generate \
    -d '{"prompt": "...", "truncate": true}'
```

---

## 7. Memory Issues

### 7.1 Out of Memory (OOM)

**Error**: `Out of memory` or kernel kills process

**Solution**:
```bash
# Check memory usage
free -h
ps aux | grep mycelium

# Reduce peer count (each peer uses ~2.5MB)
./target/release/mycelium-node --max-peers 50

# Reduce model size (use higher quantization)
# Enable gradient checkpointing
./target/release/mycelium-node --gradient-checkpointing

# Add swap space
sudo swapon --show
```

### 7.2 Memory Leak Suspected

**Symptom**: Memory usage grows continuously

**Debug**:
```bash
# Monitor memory over time
watch -n 5 'ps -o rss,vsz -p $(pgrep mycelium-node)'

# Check for known issues
# https://github.com/HautlyS/mycelium/issues

# Report if leak confirmed
# Include memory growth rate and duration
```

**Temporary workaround**:
```bash
# Restart node periodically
# Use systemd timer or cron
```

---

## 8. GPU Issues

### 8.1 CUDA Not Detected

**Error**: `No CUDA devices found` or falling back to CPU

**Checklist**:
```bash
# Is NVIDIA driver installed?
nvidia-smi

# Is CUDA Toolkit installed?
nvcc --version

# Are CUDA libraries in library path?
ldconfig -p | grep cuda

# Is the binary compiled with CUDA?
cargo build --release --features cuda
```

**Solution**:
```bash
# Install NVIDIA driver
sudo apt install nvidia-driver-535  # or latest

# Install CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# Rebuild with CUDA
cargo clean
cargo build --release --features cuda
```

### 8.2 GPU Out of Memory

**Error**: `CUDA out of memory` or `cuMemAlloc failed`

**Solution**:
```bash
# Check GPU memory
nvidia-smi

# Reduce model size
# Use Q4 instead of Q8, or Q2 instead of Q4

# Reduce context length
./target/release/mycelium-node --max-context-length 2048

# Kill other GPU processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill
```

### 8.3 Metal Not Working (macOS)

**Error**: `Metal device not available`

**Checklist**:
```bash
# Check macOS version (need 12.0+)
sw_vers

# Check chip (Apple Silicon required)
system_profiler SPHardwareDataType

# Is Metal feature enabled in build?
cargo build --release --features metal
```

---

## 9. WASM/Browser Issues

### 9.1 WebGPU Not Available

**Error**: `WebGPU not supported in this browser`

**Solution**:
- Chrome 113+: Should work out of the box
- Firefox 120+: Enable `dom.webgpu.enabled` in about:config
- Safari 16.4+: Enable WebGPU in Develop menu
- Update browser to latest version

### 9.2 WASM Module Won't Load

**Error**: `Failed to load WASM module`

**Solution**:
```javascript
// Check browser console for errors
// Common issues:
// - Incorrect MIME type (configure server to serve .wasm as application/wasm)
// - CORS issues (check Access-Control headers)
// - Memory limits (browser limits WASM to 2-4GB)

// For local development, use a proper server:
npx serve .  // or any static file server
```

### 9.3 Poor Browser Performance

**Symptom**: Very slow inference in browser

**Causes**:
- Browser memory limits
- WebGPU not using hardware acceleration
- CPU fallback is slow

**Solution**:
- Use Chrome (best WebGPU support)
- Ensure hardware acceleration is enabled
- Reduce model size for browser use
- Use desktop node for heavy inference

---

## 10. Spore Protocol Issues

### 10.1 Spore Verification Failed

**Error**: `Spore genome integrity check failed`

**Causes**:
- Download corrupted
- Malicious spore
- Transport error

**Solution**:
```bash
# Check which verification failed
# CRC32: Transport corruption — retry download
# SHA-256: Weight integrity — reject spore, report source
# Signature: Authentication — reject spore, quarantine source

# Retry spore download
# The protocol should automatically retry from different sources
```

### 10.2 Spore Not Propagating

**Symptom**: Spores not reaching other nodes

**Check**:
```bash
# Check spore status
curl http://localhost:8080/spores

# Verify network connectivity
curl http://localhost:8080/peers

# Check rate limiting
# SporeAvailable limited to 1 per 60 seconds per peer
```

---

## 11. API Issues

### 11.1 API Returns 500 Error

**Error**: `Internal Server Error`

**Solution**:
```bash
# Check logs for details
tail -f ~/.mycelium/logs/*.log

# Common causes:
# - Model not loaded
# - Compute backend error
# - Out of memory

# Restart node if necessary
pkill mycelium-node
./target/release/mycelium-node
```

### 11.2 WebSocket Connection Drops

**Symptom**: WebSocket disconnects frequently

**Causes**:
- Network instability
- Server restart
- Timeout

**Solution**:
```javascript
// Implement reconnection logic
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onclose = () => {
    setTimeout(() => {
        // Reconnect
    }, 1000);
};
```

---

## 12. Common Error Messages

### 12.1 Error Message Index

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `No CUDA devices found` | No GPU or driver issue | Check nvidia-smi, install drivers |
| `Address already in use` | Port conflict | Change ports or kill conflicting process |
| `Failed to load model` | File issue | Check path, permissions, file integrity |
| `Out of memory` | Insufficient RAM | Reduce model size, add swap |
| `Key not found in DHT` | New network or partition | Wait for DHT population |
| `Spore verification failed` | Corrupted/malicious spore | Reject, retry from different source |
| `Gradient norm exceeded` | Anomalous gradient | Reject gradient, check source node |
| `Context length exceeded` | Input too long | Reduce input or increase limit |
| `WebGPU not supported` | Browser incompatibility | Update browser, enable flags |
| `Connection refused` | Peer unreachable | Check firewall, NAT, peer status |

---

## 13. Getting Help

### 13.1 Self-H Resources

1. **This document** — Check relevant section above
2. **ARCHITECTURE.md** — Understand system design
3. **CROSSDEVICE.md** — Platform-specific issues
4. **GitHub Issues** — Search for similar problems

### 13.2 Reporting Bugs

When reporting a bug, include:

```markdown
## Environment
- OS: Ubuntu 22.04 / macOS 13 / Windows 11
- Rust: 1.75.0
- Hardware: CPU/GPU model, RAM
- Mycelium version: v0.2.0

## Issue Description
What happened and what you expected

## Steps to Reproduce
1. ...
2. ...
3. ...

## Logs
```
Relevant log output with RUST_LOG=debug
```

## Additional Context
Any other relevant information
```

### 13.3 Community Support

- **GitHub Issues**: https://github.com/HautlyS/mycelium/issues
- **GitHub Discussions**: https://github.com/HautlyS/mycelium/discussions

---

*This document grows with experience. If you encounter an issue not covered here, please contribute the solution!*

**Last Updated**: April 10, 2026
**Version**: v0.2.0

---

## See Also

- [CROSSDEVICE.md](CROSSDEVICE.md) — Cross-platform build guide
- [PERFORMANCE.md](PERFORMANCE.md) — Benchmarks and optimization
- [SECURITY.md](SECURITY.md) — Security model and threat analysis
- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and components
