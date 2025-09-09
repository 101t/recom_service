FROM rust:1.70-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN addgroup --system --gid 1000 appgroup && \
    adduser --system --uid 1000 --gid 1000 appuser

WORKDIR /app

# Copy built binary
COPY --from=builder /app/target/release/recommendation-service ./

# Create data directory with appropriate permissions
RUN mkdir /data && chown appuser:appgroup /data

USER appuser

EXPOSE 8080

ENV RUST_LOG=info
ENV DELTA_TABLE_PATH=/data/delta/tables
ENV QDRANT_URL=http://qdrant:6333
ENV EMBEDDER_MODEL=voyage-3-large

CMD ["./recommendation-service"]