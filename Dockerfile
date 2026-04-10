FROM rust:1.75-slim as builder

WORKDIR /build

COPY . .
RUN cargo build --release --workspace

FROM debian:bookworm-slim

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/target/release/mycelium-node /usr/local/bin/

EXPOSE 8080 4001

ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/mycelium-node"]
CMD ["--api-port", "8080", "--listen", "0.0.0.0:4001"]