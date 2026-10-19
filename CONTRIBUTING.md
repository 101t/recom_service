# Contributing

Thanks for helping! Bug reports, docs fixes and features are all welcome.

## Development setup

You need Rust (stable, 1.88 or newer) and Docker.

```bash
make dev        # run with no external dependencies (in-memory store + hash embedder)
make qdrant     # start Qdrant, then:
make run        # run against Qdrant with the real multilingual model
make test       # unit + API tests (no network needed)
make e2e        # build the image, start the full stack, run the black-box suite
```

Before opening a pull request, run `make ci` (format check, clippy with warnings
as errors, tests). CI also runs the e2e suite against the Docker image.

## Project layout

| Path | What lives there |
|---|---|
| `src/api/` | HTTP handlers (thin) and the OpenAPI document |
| `src/services/` | Business logic: datasets & training, recommender, prediction logs |
| `src/embedding/` | `Embedder` trait and providers (fastembed, OpenAI-compatible, Voyage, hash) |
| `src/vector_store/` | `VectorStore` trait, Qdrant and in-memory backends, MMR |
| `src/storage/` | Dataset metadata and raw data on disk; tabular parsing |
| `src/models/` | Request/response types and the filter DSL |
| `tests/api.rs` | Hermetic API tests (full HTTP stack, in-memory store, hash embedder) |
| `e2e/` | Black-box tests run against a real deployment |

## Common changes

**Add an embedding provider**: implement `embedding::Embedder` in a new module under
`src/embedding/`, add a variant to `config::EmbeddingProvider` (and its parsing in
`Settings::from_lookup`), then wire it into `embedding::build`. Apply query/document
prefixes if your model needs them (see `Prefixes`).

**Add a vector store**: implement `vector_store::VectorStore` (see `memory.rs` for a
compact reference), add a `VectorStoreKind` variant and construct it in
`AppState::initialize`. Run the API tests against it by building the state with
`AppState::with_components`.

**Add an endpoint**: write the handler in `src/api/` with a `#[utoipa::path]`
annotation and register it in `api::routes`; it appears in `/docs/` automatically.
Add a test to `tests/api.rs` (and to `e2e/` when it touches real infrastructure).

## Conventions

- Errors: return `AppError`; it maps to the JSON envelope `{"error": {"code", "message"}}`.
- Keep handlers free of business logic; services must not depend on actix types.
- Blocking or CPU-heavy work (Polars, ONNX) goes through `spawn_blocking`.
- New configuration goes into `config.rs`, `.env.example` and the README table.
- Commit messages: short imperative summary (`feat: add Cohere embedder`).
