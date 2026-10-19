# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   chef     - Rust toolchain + cargo-chef
#   planner  - computes the dependency "recipe" (changes only when Cargo.toml/lock change)
#   builder  - builds dependencies (cached layer), then the service
#   runtime  - slim Debian, non-root user, just the binary
#
# Build:  docker build -t recommendation-service .
# Bake the embedding model into the image (no download at startup, works offline):
#         docker build --build-arg PRELOAD_MODEL=true -t recommendation-service:offline .

ARG RUST_VERSION=1.99
ARG DEBIAN_VERSION=trixie

FROM rust:${RUST_VERSION}-slim-${DEBIAN_VERSION} AS chef
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef --locked --version 0.1.78
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --locked --recipe-path recipe.json
COPY . .
RUN cargo build --release --locked --bin recommendation-service \
    && cp target/release/recommendation-service /usr/local/bin/recommendation-service

FROM debian:${DEBIAN_VERSION}-slim AS runtime
ARG VERSION=dev
LABEL org.opencontainers.image.title="recommendation-service" \
      org.opencontainers.image.description="Multilingual vector recommendation service backed by Qdrant" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 recom \
    && useradd --system --uid 10001 --gid recom --home-dir /data --no-create-home --shell /usr/sbin/nologin recom \
    && mkdir -p /data /models \
    && chown recom:recom /data /models

COPY --from=builder /usr/local/bin/recommendation-service /usr/local/bin/recommendation-service

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DATA_DIR=/data \
    EMBEDDING_CACHE_DIR=/models \
    LOG_FORMAT=json \
    RUST_LOG=info

USER recom
WORKDIR /data

# Optionally download the embedding model at build time (adds ~450 MB for the default model).
ARG PRELOAD_MODEL=false
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
        EMBEDDING_MODEL="$EMBEDDING_MODEL" recommendation-service download-model; \
    fi

VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD ["recommendation-service", "healthcheck"]

ENTRYPOINT ["recommendation-service"]
CMD ["serve"]
