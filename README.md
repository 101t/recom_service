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
cp .env.example .env   # if you have one, otherwise create .env
```

Key variables (all optional – sensible defaults are used):

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant gRPC/HTTP endpoint |
| `QDRANT_COLLECTION` | `recommendations` | Default collection name |
| `QDRANT_VECTOR_SIZE` | `384` | Embedding vector dimension |
| `USE_LOCAL_EMBEDDER` | `true` | `true` → local FastEmbed, `false` → Voyage AI |
| `LOCAL_EMBEDDER_URL` | `http://localhost:8001` | FastEmbed service URL |
| `VOYAGE_API_KEY` | – | Required when `USE_LOCAL_EMBEDDER=false` |
| `DATASET_STORAGE_PATH` | `./data/datasets` | Where Parquet datasets are stored |
| `PREDICTION_LOG_PATH` | `./data/predictions` | Where prediction logs are written |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |

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
| `POST` | `/api/v1/datasets` | Upload a dataset (multipart). Query params: `background`, `dataset_id`, `text_column` |
| `GET` | `/api/v1/datasets` | List stored datasets |
| `GET` | `/api/v1/datasets/{id}` | Get dataset info |
| `DELETE` | `/api/v1/datasets/{id}` | Delete a dataset and its vectors |
| `GET` | `/api/v1/datasets/jobs/{job_id}` | Poll background upload job status |

### Recommendations

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/recommendations` | Search for similar items |

**Request body** supports the full Qdrant filter DSL:

```json
{
  "query": "comfortable running shoes",
  "top_k": 10,
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

```bash
cargo test
```

Unit tests cover the Qdrant client, filter serialization, dataset handler, prediction logger, and API response structures.

## Project Structure

```
src/
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
