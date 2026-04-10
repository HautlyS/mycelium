# Mycelium Docker Image
# Build: docker build -t mycelium .
# Run:  docker run -p 8080:8080 -p 4001:4001 mycelium
# Or:  docker run -d -p 8080:8080 -p 4001:4001 --network host mycelium

# Multi-stage build for smaller image
FROM rust:1.75-slim as builder

WORKDIR /build

# Copy only files needed for build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY shaders ./shaders

# Build the binary
RUN cargo build --release --workspace

# Production image
FROM debian:bookworm-slim

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/false mycelium

# Copy binary from builder
COPY --from=builder /build/target/release/mycelium-node /usr/local/bin/

# Create data directory
RUN mkdir -p /data && chown mycelium:mycelium /data

# Switch to non-root user
USER mycelium

# Expose ports
EXPOSE 8080 4001

# Set environment
ENV MYCELIUM_DATA_DIR=/data
ENV RUST_LOG=info

# Default run options
VOLUME ["/data"]

ENTRYPOINT ["/usr/local/bin/mycelium-node"]
CMD ["--api-port", "8080", "--listen", "0.0.0.0:4001", "--data-dir", "/data"]