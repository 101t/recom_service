SHELL := /bin/bash
.DEFAULT_GOAL := help

IMAGE         ?= recommendation-service
TAG           ?= local
PLATFORMS     ?= linux/amd64,linux/arm64
PRELOAD_MODEL ?= false
COMPOSE       ?= docker compose
BASE_URL      ?= http://localhost:8000

export RECOM_IMAGE := $(IMAGE):$(TAG)

.PHONY: help run dev qdrant fmt fmt-check lint test check build doc ci \
        up down logs ps e2e e2e-run docker-build docker-buildx download-model clean clean-docker

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Development ──────────────────────────────────────────────────────────────

run: ## Run the service locally (needs Qdrant: `make qdrant`)
	cargo run --release -- serve

dev: ## Run without external dependencies (in-memory store + hash embedder)
	VECTOR_STORE=memory EMBEDDING_PROVIDER=hash cargo run -- serve

qdrant: ## Start only Qdrant (ports 6333/6334)
	$(COMPOSE) up -d --wait qdrant

download-model: ## Download the configured embedding model into the local cache
	cargo run --release -- download-model

# ── Quality ──────────────────────────────────────────────────────────────────

fmt: ## Format the code
	cargo fmt --all

fmt-check: ## Check formatting
	cargo fmt --all -- --check

lint: ## Clippy with warnings as errors
	cargo clippy --workspace --all-targets --locked -- -D warnings

test: ## Unit + API tests (hermetic: no Qdrant, no model download)
	cargo test --locked

check: ## Type-check, including the build without the fastembed feature
	cargo check --workspace --all-targets --locked
	cargo check --locked --no-default-features

ci: fmt-check lint test ## Everything CI runs except e2e

build: ## Release build
	cargo build --release --locked

doc: ## Build and open the Rust API docs
	cargo doc --no-deps --open

# ── Docker ───────────────────────────────────────────────────────────────────

up: ## Build the image and start Qdrant + the service
	$(COMPOSE) up -d --build --wait

down: ## Stop the stack (keeps data volumes)
	$(COMPOSE) down

logs: ## Follow service logs
	$(COMPOSE) logs -f api

ps: ## Show stack status
	$(COMPOSE) ps

e2e: ## Build, start the stack and run the e2e suite (E2E_KEEP=1 keeps it running)
	$(COMPOSE) up -d --build --wait
	@status=0; E2E_BASE_URL=$(BASE_URL) cargo test -p recom-e2e --locked || status=$$?; \
	if [ $$status -ne 0 ]; then $(COMPOSE) logs --no-color --tail=200 api; fi; \
	if [ -z "$(E2E_KEEP)" ]; then $(COMPOSE) down; fi; \
	exit $$status

e2e-run: ## Run the e2e suite against an already running service (BASE_URL=...)
	E2E_BASE_URL=$(BASE_URL) cargo test -p recom-e2e --locked

docker-build: ## Build the image for the local platform (PRELOAD_MODEL=true bakes the model in)
	docker build --build-arg PRELOAD_MODEL=$(PRELOAD_MODEL) -t $(IMAGE):$(TAG) .

docker-buildx: ## Build a multi-arch image with buildx (add PUSH=1 to push)
	docker buildx build --platform $(PLATFORMS) --build-arg PRELOAD_MODEL=$(PRELOAD_MODEL) \
		-t $(IMAGE):$(TAG) $(if $(PUSH),--push,) .

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts
	cargo clean

clean-docker: ## Stop the stack and delete its volumes (datasets, vectors, model cache)
	$(COMPOSE) down -v
