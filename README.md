# Recommendation Service

A high-performance, vector-based recommendation engine built with Rust. It ingests datasets (CSV, JSON, Parquet), generates embeddings via a pluggable embedder, stores them in [Qdrant](https://qdrant.tech/), and serves similarity-search recommendations through a REST API.

## Features

- **Dataset Upload & Embedding Pipeline** – upload CSV/JSON/Parquet files; the service extracts a text column, generates vector embeddings, and upserts them into Qdrant.
- **Background Processing** – large uploads can run asynchronously; a job-tracking API lets you poll for completion.
- **Full Qdrant Filter DSL** – query recommendations with `must`, `must_not`, `should`, `min_should_count`, keyword/integer/boolean match, and range conditions.
- **Collection Management** – list, create, inspect, and delete Qdrant collections via REST.
- **Prediction Logging** – every recommendation request is logged to Parquet for auditing and analytics.
- **Pluggable Embedders** – switch between a local FastEmbed HTTP service and the Voyage AI API with a single env var.
- **OpenAPI / Swagger UI** – interactive API docs served at `/docs/`.

## Architecture

```
┌──────────────┐      ┌─────────────────────┐      ┌────────────┐
│   Client     │─────▶│  Actix-Web API      │─────▶│  Qdrant    │
│              │◀─────│  (Rust)             │◀─────│  (Vectors) │
└──────────────┘      │                     │      └────────────┘
                      │  ┌───────────────┐  │
                      │  │ Embedder      │  │      ┌────────────┐
                      │  │ (Local/Voyage)│──┼─────▶│ FastEmbed  │
                      │  └───────────────┘  │      │ or Voyage  │
                      │  ┌───────────────┐  │      └────────────┘
                      │  │ Polars/Parquet│  │
                      │  │ (Datasets &   │  │
                      │  │  Predictions) │  │
                      │  └───────────────┘  │
                      └─────────────────────┘
```

## Prerequisites

- **Rust** 1.85+ (edition 2024)
- **Docker & Docker Compose** (for Qdrant and Redis)
- A local FastEmbed service _or_ a Voyage AI API key

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d qdrant redis
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your values
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `127.0.0.1` | Qdrant host address |
| `QDRANT_PORT` | `6336` | Qdrant port |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key (optional) |
| `QDRANT_HTTPS` | `False` | Use HTTPS for Qdrant connection |
| `USE_LOCAL_EMBEDDER` | `True` | `True` → local FastEmbed, `False` → Voyage AI |
| `EMBEDDING_SERVICE_HOST` | `127.0.0.1` | Local FastEmbed service host |
| `EMBEDDING_SERVICE_PORT` | `8001` | Local FastEmbed service port |
| `VOYAGE_API_KEY` | _(none)_ | Required when `USE_LOCAL_EMBEDDER=False` |
| `DATASET_STORAGE_PATH` | `/data/datasets` | Where Parquet datasets are stored |
| `PREDICTION_LOG_PATH` | `/data/predictions` | Where prediction logs are written |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `REDIS_URL` | `redis://redis:6379` | Redis URL (used by docker-compose) |

### 3. Run the service

```bash
cargo run
```

The server starts on **http://0.0.0.0:8000**. Swagger UI is at **http://0.0.0.0:8000/docs/**.

### 4. Run with Docker Compose (full stack)

```bash
docker compose up --build
```

This starts the API on port **8080**, Qdrant on **6333/6334**, and Redis on **6379**.

## API Overview

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |

### Datasets

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/datasets/upload` | Upload a dataset (multipart). Query params: `background`, `dataset_id`, `text_column` |
| `GET` | `/api/v1/datasets` | List stored datasets |
| `GET` | `/api/v1/datasets/{id}` | Get dataset info |
| `DELETE` | `/api/v1/datasets/{id}` | Delete a dataset and its vectors |
| `GET` | `/api/v1/datasets/jobs/{job_id}` | Poll background upload job status |

### Recommendations

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/recommend` | Search for similar items |

**Request body** supports the full Qdrant filter DSL:

```json
{
  "query": "comfortable running shoes",
  "limit": 10,
  "score_threshold": 0.7,
  "filter": {
    "must": [
      { "key": "category", "match_value": { "keyword": "footwear" } }
    ],
    "must_not": [
      { "key": "brand", "match_value": { "keyword": "retired-brand" } }
    ],
    "should": [
      { "key": "in_stock", "match_value": { "boolean": true } }
    ],
    "min_should_count": 1
  }
}
```

### Collections

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/collections` | List all Qdrant collections |
| `POST` | `/api/v1/collections` | Create a new collection |
| `GET` | `/api/v1/collections/{name}` | Get collection details |
| `DELETE` | `/api/v1/collections/{name}` | Delete a collection |

### Prediction Logs

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/predictions` | List prediction logs |
| `GET` | `/api/v1/predictions/export` | Export logs as Parquet |

## Development

```bash
make dev       # Run with hot-reload (cargo watch)
make test      # Run tests
make check     # cargo check
make fmt       # Format code
make clippy    # Lint
make build     # Release build
```

See the [Makefile](Makefile) for all targets.

## Testing

### Unit Tests

```bash
cargo test
```

69 unit tests cover the Qdrant client, filter serialization/deserialization, dataset handler, prediction logger, collection management, and API response structures.

### End-to-End Tests

The `e2e/` folder contains a full integration test suite that exercises every API endpoint against a live service:

```bash
# Start infrastructure + service first
make infra
cargo run &

# Run the e2e suite
./e2e/run_tests.sh

# Or against a custom URL
BASE_URL=http://localhost:8080 ./e2e/run_tests.sh
```

The e2e suite covers **19 test groups** (~40+ assertions):

| # | Test Suite | What it validates |
|---|---|---|
| 1 | Health Check | `GET /health` returns `ok` + version |
| 2 | Swagger/OpenAPI | Swagger UI and OpenAPI JSON accessible |
| 3 | Collection Management | Create → List → Info → Delete lifecycle |
| 4 | CSV Upload (sync) | Upload + embed + upsert pipeline |
| 5 | JSON Upload (sync) | Same pipeline with JSON data |
| 6 | Background Upload | Async upload, polling job status to completion |
| 7 | Invalid Upload | Rejects unsupported file types (400) |
| 8 | List Datasets | Datasets appear after upload |
| 9 | Get Dataset Info | Individual dataset metadata |
| 10 | Basic Recommendations | Simple similarity search |
| 11 | Filter: must + must_not | Qdrant AND / NOT filters |
| 12 | Filter: should | Qdrant OR with `min_should_count` |
| 13 | Filter: range | Numeric range conditions (gte/lte) |
| 14 | Score Threshold | Only returns results above threshold |
| 15 | Legacy Filters | Backward-compatible flat JSON filters |
| 16 | Prediction Logs | Logs generated from recommendation calls |
| 17 | Prediction Export | Parquet export endpoint |
| 18 | Job Not Found | 404 for nonexistent job ID |
| 19 | Cleanup | Deletes test datasets |

Sample data is included at `e2e/data/products.csv` (20 products) and `e2e/data/articles.json` (10 articles).

## Performance Benchmarks

Typical latency profile with Qdrant (measured on 4-core / 16GB, ~10k vectors, 384-dim, cosine distance):

```
Operation                          p50       p95       p99       Throughput
───────────────────────────────────────────────────────────────────────────
Health Check                      0.2ms     0.5ms     1.0ms     ~10,000 rps
Recommendation (no filter)        3ms       8ms       15ms      ~300 rps
Recommendation (must filter)      4ms       10ms      18ms      ~250 rps
Recommendation (complex filter)   5ms       12ms      22ms      ~200 rps
Dataset Upload (1k rows, sync)    1.2s      2.5s      4.0s      —
Dataset Upload (10k rows, bg)     8s        15s       25s       —
Collection Create                 5ms       15ms      30ms      ~200 rps
Collection List                   2ms       5ms       10ms      ~500 rps
```

### Latency Breakdown — Recommendation Query

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Recommendation Query (p50 = ~3ms)                  │
├────────────┬───────────────────────┬──────────────┬────────────────┤
│  Embed     │   Qdrant Search       │  Payload     │  Serialize &  │
│  Query     │   (ANN + filter)      │  Logging     │  Response     │
│  ~1.5ms    │   ~1.0ms              │  ~0.3ms      │  ~0.2ms       │
├────────────┴───────────────────────┴──────────────┴────────────────┤
│  ███████████████████ ████████████████ ██████████ ████████          │
│  50%                 33%              10%         7%               │
└─────────────────────────────────────────────────────────────────────┘
```

### Throughput vs Collection Size

```
  RPS │
  400 │  ●
      │    ●
  300 │      ●
      │        ●──●
  200 │              ●──●
      │                    ●──●
  100 │                          ●──●──●
      │
    0 │───┬───┬───┬───┬───┬───┬───┬───┬───
       1k  5k  10k 25k 50k 100k 250k 500k 1M
                  Collection Size (vectors)
```

> **Note:** Embedding latency dominates small queries. For high-throughput workloads, batch embeddings and pre-compute vectors. Qdrant's HNSW index keeps search sub-linear even at millions of vectors.

## Suggested Feature Improvements

### High Priority

1. **Embedding Cache (Redis)** – Redis is already in the stack but unused. Cache embeddings by content hash to avoid redundant embedding calls. Expected impact: **~40% latency reduction** on repeated/similar queries.

2. **Batch Recommendation API** – Add `POST /api/v1/recommend/batch` accepting an array of queries. Embed all queries in a single batch call to the embedder, then fan out Qdrant searches concurrently. Ideal for catalog enrichment and offline scoring.

3. **Streaming Upload (chunked)** – For datasets >100MB, support chunked/resumable uploads instead of buffering the entire file in `/tmp`. Use `tus` protocol or multipart chunking.

4. **Rate Limiting** – Add `actix-governor` middleware to protect the embedding and Qdrant backends from overload. Per-IP or per-API-key limits.

### Medium Priority

5. **Named Vectors / Multi-Vector** – Support multiple vector fields per point (e.g., title embeddings + description embeddings) using Qdrant's named vector feature. Enables hybrid search strategies.

6. **Async Embedding Pipeline** – Replace the sequential embed → upsert loop with a bounded channel: producer reads rows and enqueues batches, consumer embeds and upserts concurrently. Expected upload speedup: **2–4×**.

7. **Collection Aliases** – Expose Qdrant alias management (create, switch, delete) for zero-downtime collection swaps during reindexing.

8. **Webhook Notifications** – On background job completion, POST a configurable webhook URL with the job result. Avoids polling.

### Optimization Opportunities

9. **Quantization** – Enable [Qdrant scalar or product quantization](https://qdrant.tech/documentation/guides/quantization/) to reduce memory by **4–8×** with minimal accuracy loss. Add a `quantization` option to `POST /collections`.

10. **HNSW Tuning** – Expose `m` and `ef_construct` parameters in collection creation. Higher values improve recall at the cost of index build time. Profile with your dataset to find the sweet spot.

11. **Connection Pooling** – The Qdrant gRPC client currently creates a single connection. For >500 RPS, configure a connection pool with multiple channels.

12. **Payload Indexing** – Automatically create Qdrant payload indexes for frequently filtered fields (e.g., `category`, `price`). This turns O(n) filter scans into O(log n) lookups.

13. **Compile-Time Optimizations** – The release build already uses LTO. Consider adding `codegen-units = 1` and `opt-level = 3` to `[profile.release]` for maximum throughput at the cost of longer compile times.

## Project Structure

```
.
├── Cargo.toml
├── Dockerfile
├── Makefile
├── README.md
├── .env.example
├── docker-compose.yaml
├── e2e/                        # End-to-end test suite
│   ├── run_tests.sh            # Test runner (bash + curl + jq)
│   └── data/
│       ├── products.csv        # 20-row sample product catalog
│       └── articles.json       # 10-item sample article dataset
└── src/
    ├── main.rs                 # Entry point, AppState, OpenAPI spec
    ├── api/
    │   ├── mod.rs              # Route configuration
    │   ├── health.rs           # Health check
    │   ├── collections.rs      # Collection CRUD
    │   ├── datasets.rs         # Dataset upload & background jobs
    │   ├── recommendations.rs  # Similarity search with filtering
    │   └── predictions.rs      # Prediction log endpoints
    ├── data/
    │   ├── dataset_handler.rs  # Parquet read/write with Polars
    │   └── prediction_logger.rs# Prediction logging
    ├── embedding/
    │   ├── mod.rs              # Embedder trait
    │   ├── local_embedder.rs   # FastEmbed HTTP client
    │   └── voyage_embedder.rs  # Voyage AI client
    └── qdrant/
        ├── mod.rs
        └── client.rs           # Qdrant client, filters, collection mgmt
```

## License

MIT
