.PHONY: run dev test e2e check build release fmt clippy clean docker-up docker-down docker-build infra

# ── Development ───────────────────────────────────────────────

## Run the service (debug)
run:
	cargo run

## Run with auto-reload (requires cargo-watch: cargo install cargo-watch)
dev:
	cargo watch -x run

## Run all tests
test:
	cargo test

## Run end-to-end tests (requires running service + Qdrant)
e2e:
	./e2e/run_tests.sh

## Type-check without building
check:
	cargo check

## Format code
fmt:
	cargo fmt

## Lint with clippy
clippy:
	cargo clippy -- -D warnings

## Build release binary
build:
	cargo build --release

## Full CI check: fmt + clippy + tests
ci: fmt clippy test

# ── Docker ────────────────────────────────────────────────────

## Start the full stack (API + Qdrant + Redis)
docker-up:
	docker compose up --build -d

## Stop all containers
docker-down:
	docker compose down

## Build Docker image only
docker-build:
	docker build -t recommendation-service .

## Start only infrastructure (Qdrant + Redis)
infra:
	docker compose up -d qdrant redis

# ── Cleanup ───────────────────────────────────────────────────

## Remove build artifacts
clean:
	cargo clean

# ── Help ──────────────────────────────────────────────────────

## Show this help
help:
	@echo "Available targets:"
	@echo "  run          Run the service (debug)"
	@echo "  dev          Run with auto-reload (cargo-watch)"
	@echo "  test         Run all unit tests"
	@echo "  e2e          Run end-to-end tests"
	@echo "  check        Type-check without building"
	@echo "  fmt          Format code"
	@echo "  clippy       Lint with clippy"
	@echo "  build        Release build"
	@echo "  ci           fmt + clippy + tests"
	@echo "  docker-up    Start full stack"
	@echo "  docker-down  Stop all containers"
	@echo "  docker-build Build Docker image"
	@echo "  infra        Start Qdrant + Redis only"
	@echo "  clean        Remove build artifacts"
