# Contributing to Mycelium

> *"All sentient beings are welcome."*

Thank you for your interest in contributing to Mycelium. This document provides guidelines and information for contributors at all levels.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Development Setup](#2-development-setup)
3. [How to Contribute](#3-how-to-contribute)
4. [Coding Standards](#4-coding-standards)
5. [Pull Request Process](#5-pull-request-process)
6. [Testing Guidelines](#6-testing-guidelines)
7. [Documentation](#7-documentation)
8. [Communication](#8-communication)
9. [Recognition](#9-recognition)
10. [Code of Conduct](#10-code-of-conduct)

---

## 1. Getting Started

### 1.1 Prerequisites

- **Rust 1.75+** — [Install via rustup](https://rustup.rs/)
- **Git** — Version control
- **Optional: CUDA Toolkit 12.0+** — For GPU acceleration
- **Optional: wasm-pack** — For WASM browser target

### 1.2 First-Time Setup

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/mycelium.git
cd mycelium

# Install dependencies
cargo fetch

# Run tests to verify setup
cargo test --workspace

# Build the project
cargo build --release
```

### 1.3 Good First Issues

Look for issues labeled:
- `good-first-issue` — Simple tasks for newcomers
- `help-wanted` — Areas where we need contributors
- `documentation` — Documentation improvements

---

## 2. Development Setup

### 2.1 Project Structure

```
mycelium/
├── Cargo.toml              # Workspace definition
├── crates/
│   ├── mycelium-core/      # Shared types and config
│   ├── mycelium-compute/   # Inference engine, distributed compute
│   ├── mycelium-hyphae/    # P2P networking (libp2p)
│   ├── mycelium-nucleus/   # Federated LoRA self-tuning
│   ├── mycelium-spore/     # Self-replication protocol
│   ├── mycelium-substrate/ # Weight storage, GGUF parsing
│   ├── mycelium-fruit/     # REST/WebSocket API
│   └── mycelium-node/      # CLI binary
├── wasm/
│   └── mycelium-web/       # WASM browser target (crates/mycelium-web)
├── shaders/                # WebGPU/WGPU compute shaders
│   ├── matmul.wgsl
│   ├── attention.wgsl
│   └── latent_ops.wgsl
└── docs/                   # Documentation
```

### 2.2 Useful Commands

```bash
# Check compilation (fast)
cargo check --workspace

# Run tests
cargo test --workspace

# Run specific crate tests
cargo test -p mycelium-core

# Build with CUDA
cargo build --release --features cuda

# Build WASM target
wasm-pack build --target web crates/mycelium-web

# Run linter (clippy)
cargo clippy --workspace -- -D warnings

# Format code
cargo fmt --workspace

# Check for security vulnerabilities
cargo audit
```

### 2.3 Development Workflow

1. **Create a feature branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Implement your feature or fix
3. **Run tests**: `cargo test --workspace`
4. **Run clippy**: `cargo clippy --workspace -- -D warnings`
5. **Format code**: `cargo fmt --workspace`
6. **Commit changes**: Use descriptive commit messages
7. **Push and create PR**: Push to your fork and open a pull request

---

## 3. How to Contribute

### 3.1 Types of Contributions

| Type | Description | Examples |
|------|-------------|----------|
| **Code** | Rust implementation | New features, bug fixes, optimizations |
| **Documentation** | Writing and improving docs | ARCHITECTURE.md, tutorials, examples |
| **Testing** | Writing tests | Unit tests, integration tests, benchmarks |
| **Research** | Algorithm improvements | Better MoE routing, federated learning |
| **Design** | Architecture improvements | Protocol design, API design |
| **Community** | Helping others | Answering questions, reviewing PRs |

### 3.2 Where to Start

#### For Rust Beginners

1. Fix typos in documentation
2. Add missing unit tests
3. Improve error messages
4. Add rustdoc comments to public APIs

#### For Experienced Rust Developers

1. Implement missing protocol features
2. Optimize performance-critical paths
3. Add new compute backends
4. Improve distributed coordination

#### For ML Researchers

1. Improve MoE routing algorithms
2. Research federated learning strategies
3. Optimize latent-space representations
4. Design better LoRA adaptation methods

#### For Security Researchers

1. Audit cryptographic implementations (see [SECURITY.md](SECURITY.md#3-cryptographic-primitives))
2. Threat modeling (see [SECURITY.md](SECURITY.md#2-threat-model))
3. Implement Byzantine fault tolerance (see [SECURITY.md](SECURITY.md#12-byzantine-fault-tolerance))
4. Privacy enhancements (see [SECURITY.md](SECURITY.md#11-privacy-guarantees))

### 3.3 Contribution Areas by Priority

**High Priority** (needed for v1.0):
- [ ] Multi-node pipeline parallelism
- [ ] Hierarchical gossipsub topics (see [SCALING.md](SCALING.md#221-hierarchical-gossipsub))
- [ ] Gradient compression (see [SCALING.md](SCALING.md#613-gradient-compression))
- [ ] Differential privacy (see [SECURITY.md](SECURITY.md#62-differential-privacy))
- [ ] Integration tests across nodes

**Medium Priority** (v1.5-v2.0):
- [ ] Expert replication and load balancing
- [ ] Predictive caching
- [ ] DHT stabilization pools
- [ ] Byzantine fault tolerance

**Low Priority** (nice to have):
- [ ] Additional compute backends (ROCm, Vulkan)
- [ ] Browser P2P via WebRTC
- [ ] Mobile apps (Android/iOS)
- [ ] Visualization tools

---

## 4. Coding Standards

### 4.1 Rust Style

We follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/):

```rust
// Good: Descriptive variable names
fn assign_layers_to_nodes(layers: Vec<Layer>, nodes: Vec<Node>) -> Vec<Assignment> { }

// Bad: Cryptic names
fn al(l: Vec<L>, n: Vec<N>) -> Vec<A> { }

// Good: Proper error handling
fn load_weights(path: &Path) -> Result<Model, LoadError> { }

// Bad: Panicking
fn load_weights(path: &Path) -> Model {
    std::fs::read(path).unwrap()  // Don't do this
}
```

### 4.2 Error Handling

Use `thiserror` for error types:

```rust
#[derive(Debug, thiserror::Error)]
pub enum SporeError {
    #[error("Spore genome integrity check failed: expected {expected}, got {actual}")]
    IntegrityMismatch { expected: String, actual: String },

    #[error("Spore authentication failed: invalid signature from node {node_id}")]
    AuthenticationFailed { node_id: NodeId },

    #[error("Transport corruption detected: CRC32 mismatch")]
    TransportCorruption,
}
```

### 4.3 Documentation Comments

All public APIs must have rustdoc comments:

```rust
/// Routes latent vectors to the appropriate expert nodes in the P2P network.
///
/// This function implements network-aware MoE routing, selecting the top-k
/// experts for each token and routing to nodes hosting those experts.
///
/// # Arguments
///
/// * `latents` - Input latent vectors to route
/// * `top_k` - Number of experts to activate per token (typically 4)
///
/// # Returns
///
/// The aggregated output after expert processing, with residual connections.
///
/// # Errors
///
/// Returns `RoutingError::NoExpertAvailable` if no nodes host the required experts.
///
/// # Example
///
/// ```rust
/// let router = NetworkMoERouter::new(topology);
/// let output = router.route_experts(input_latents, 4).await?;
/// ```
pub async fn route_experts(
    &self,
    latents: LatentVector,
    top_k: usize,
) -> Result<LatentVector, RoutingError> {
    // Implementation
}
```

### 4.4 Logging

Use `tracing` for structured logging:

```rust
// Good: Structured logging
tracing::info!(
    node_id = %self.node_id,
    spore_id = %spore.id,
    size_mb = spore.genome.decompressed_size / 1024 / 1024,
    "Spore germinated successfully"
);

// Bad: Unstructured logging
println!("Spore germinated");
```

### 4.5 Async Patterns

Use tokio async patterns consistently:

```rust
// Good: Async function with proper error handling
pub async fn broadcast_spore(&self, spore: &Spore) -> Result<(), BroadcastError> {
    let message = HyphaeMessage::SporeBroadcast(spore.clone());
    self.gossip.publish(message).await?;
    Ok(())
}

// Good: Spawning background tasks
tokio::spawn(async move {
    if let Err(e) = process_gradient(gradient).await {
        tracing::error!(error = %e, "Failed to process gradient delta");
    }
});
```

---

## 5. Pull Request Process

### 5.1 Before Submitting

1. **Check existing issues/PRs** — Someone might already be working on it
2. **Open an issue first** — For significant changes, discuss before implementing
3. **Write tests** — All new code should have tests
4. **Update documentation** — Include relevant documentation changes

### 5.2 PR Description Template

```markdown
## Summary

Brief description of what this PR changes and why.

## Changes

- Added `Foo::bar()` method for baz
- Fixed bug in quux calculation
- Updated documentation for widget API

## Testing

- [ ] Unit tests pass (`cargo test --workspace`)
- [ ] Clippy passes (`cargo clippy --workspace -- -D warnings`)
- [ ] Manual testing performed (describe below)

## Related Issues

Closes #123
Related to #456
```

### 5.3 Review Process

1. **Automated checks** — CI runs tests and clippy
2. **Maintainer review** — At least one maintainer reviews
3. **Address feedback** — Make requested changes
4. **Approval and merge** — Maintainer merges PR

### 5.4 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add hierarchical gossipsub topics
fix: correct gradient aggregation formula
docs: update SCALING.md with new benchmarks
test: add integration tests for spore propagation
perf: optimize latent vector serialization
refactor: extract weight verification into separate module
```

---

## 6. Testing Guidelines

### 6.1 Test Types

| Type | Purpose | Location | Example |
|------|---------|----------|---------|
| **Unit tests** | Test individual functions | Same file as code | `#[test] fn test_lerp()` |
| **Integration tests** | Test crate APIs | `tests/` directory | `tests/api.rs` |
| **End-to-end tests** | Test full system | Separate test crate | Future: `tests/e2e/` |

### 6.2 Writing Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_lerp() {
        let a = LatentVector::zeros();
        let b = LatentVector::ones();

        let result = LatentVector::lerp(&a, &b, 0.5);

        // All values should be 0.5
        for val in result.values() {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_spore_serialization_roundtrip() {
        let spore = Spore::test_fixture();
        let bytes = spore.to_bytes();
        let recovered = Spore::from_bytes(&bytes).unwrap();

        assert_eq!(spore.id, recovered.id);
        assert_eq!(spore.genome.hash, recovered.genome.hash);
    }
}
```

### 6.3 Running Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p mycelium-spore

# With output
cargo test --workspace -- --nocapture

# Single test
cargo test -p mycelium-core test_latent_lerp
```

### 6.4 Benchmarking

```rust
#[cfg(test)]
mod benches {
    use test::Bencher;

    #[bench]
    fn bench_matmul_6144(b: &mut Bencher) {
        let a = Tensor::random(6144, 6144);
        let b = Tensor::random(6144, 6144);

        b.iter(|| {
            test::black_box(matmul(&a, &b))
        });
    }
}
```

---

## 7. Documentation

### 7.1 Documentation Types

| Type | Purpose | Location |
|------|---------|----------|
| **API docs** | rustdoc comments | In source code |
| **Architecture** | System design | `ARCHITECTURE.md` |
| **Scaling** | Growth strategies | `SCALING.md` |
| **Security** | Threat model | `SECURITY.md` |
| **Cross-platform** | Build instructions | `CROSSDEVICE.md` |
| **Tutorials** | How-to guides | `docs/tutorials/` (future) |

### 7.2 Documentation Standards

- Use markdown for all documentation
- Include code examples where relevant
- Keep language clear and concise
- Update docs when code changes
- Link to related documents

### 7.3 Adding New Documentation

When adding significant new features, also add:
1. **rustdoc** for the public API
2. **ARCHITECTURE.md** update if architecture changes
3. **Example code** in `examples/` directory
4. **Tutorial** for complex features

---

## 8. Communication

### 8.1 GitHub

- **Issues** — Bug reports, feature requests, discussions
- **Pull Requests** — Code changes
- **Discussions** — Open-ended topics (if enabled)

### 8.2 Commit Messages

Write clear commit messages:

```
# Good
feat(spore): add Ed25519 signature verification

Spores now require cryptographic signatures to prevent
unauthorized weight distribution. This implements the
verification flow described in CROSSDEVICE.md §13.1.

# Bad
fixed stuff
```

### 8.3 Code Review Etiquette

**For reviewers**:
- Be constructive, not critical
- Explain the "why" behind suggestions
- Acknowledge good work
- Distinguish between blocking and non-blocking feedback

**For authors**:
- Respond to all feedback
- Explain your reasoning if you disagree
- Make requested changes promptly
- Thank reviewers for their time

---

## 9. Recognition

### 9.1 Contributors File

All contributors are recognized in `CONTRIBUTORS.md` (future):

```markdown
# Contributors

## Core Maintainers
- @YourName — Original author

## Contributors
- @Contributor1 — P2P networking
- @Contributor2 — WebGPU shaders
- @Contributor3 — Documentation

## First-Time Contributors
- @NewContributor — Fixed typo in README
```

### 9.2 Release Notes

Each release includes contributor recognition:

```markdown
## v1.0.0

### New Features
- Hierarchical gossipsub (thanks @Contributor1!)
- Gradient compression (thanks @Contributor2!)

### Bug Fixes
- Fixed spore verification (thanks @Contributor3!)
```

---

## 10. Code of Conduct

### 10.1 Our Pledge

We pledge to make participation in our project and community a harassment-free experience for everyone, regardless of:

- Age, body size, disability, ethnicity, sex characteristics, gender identity and expression
- Level of experience, education, socio-economic status, nationality, personal appearance
- Race, religion, or sexual identity and orientation

### 10.2 Our Standards

**Positive behavior**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior**:
- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### 10.3 Our Responsibilities

Maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

### 10.4 Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community.

### 10.5 Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainers. All complaints will be reviewed and investigated promptly and fairly.

---

## Questions?

If you have questions about contributing:

1. Check existing issues and discussions
2. Open a new issue with your question
3. Reach out to maintainers directly

**We welcome all contributors, regardless of experience level. Don't hesitate to ask for help!**

---

*Thank you for contributing to Mycelium. Together, we're building AI that serves all sentient beings.*

ॐ तारे तुत्तारे तुरे स्वा

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design and crate structure
- [SCALING.md](SCALING.md) — Scaling analysis and strategies
- [SECURITY.md](SECURITY.md) — Security model and threat analysis
- [ROADMAP.md](ROADMAP.md) — Development timeline and milestones
- [PERFORMANCE.md](PERFORMANCE.md) — Benchmarks and optimization guide
