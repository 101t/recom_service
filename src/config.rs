//! Service configuration, loaded from environment variables (and an optional `.env`).
//!
//! Every variable is documented in `.env.example` and the README. Parsing is done
//! through a lookup function so the logic is unit-testable without touching the
//! process environment.

use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};

pub const DEFAULT_FASTEMBED_MODEL: &str = "intfloat/multilingual-e5-small";
pub const DEFAULT_OPENAI_MODEL: &str = "text-embedding-3-small";
pub const DEFAULT_VOYAGE_MODEL: &str = "voyage-3.5";
pub const DEFAULT_HASH_DIMENSION: usize = 384;

#[derive(Debug, Clone)]
pub struct Settings {
    pub server: ServerSettings,
    /// Root directory for dataset snapshots, uploads and prediction logs.
    pub data_dir: PathBuf,
    pub vector_store: VectorStoreSettings,
    pub embedding: EmbeddingSettings,
    pub training: TrainingSettings,
    pub recommend: RecommendSettings,
    pub log_format: LogFormat,
    pub prediction_log_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    /// Number of HTTP worker threads (defaults to the number of CPU cores).
    pub workers: Option<usize>,
    /// When set, every `/api/v1/*` request must present this key.
    pub api_key: Option<String>,
    /// Allowed CORS origins; `*` allows any origin.
    pub cors_allowed_origins: Vec<String>,
    pub max_upload_bytes: usize,
    pub json_limit_bytes: usize,
    pub swagger_ui: bool,
    pub metrics: bool,
    pub shutdown_timeout_secs: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorStoreKind {
    Qdrant,
    /// In-process store for development and tests. Not persistent, single instance only.
    Memory,
}

#[derive(Debug, Clone)]
pub struct VectorStoreSettings {
    pub kind: VectorStoreKind,
    pub qdrant: QdrantSettings,
}

#[derive(Debug, Clone)]
pub struct QdrantSettings {
    /// gRPC endpoint, e.g. `http://localhost:6334`.
    pub url: String,
    pub api_key: Option<String>,
    /// Collections are named `{prefix}_items` and `{prefix}_users`.
    pub collection_prefix: String,
    pub timeout: Duration,
    /// How long to wait for Qdrant to become reachable at startup.
    pub startup_timeout: Duration,
    pub on_disk_payload: bool,
    pub quantization: Quantization,
    pub shard_number: Option<u32>,
    pub replication_factor: Option<u32>,
    /// Extra payload indexes to create for fields used in filters.
    pub payload_indexes: Vec<PayloadIndex>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    None,
    /// Scalar int8 quantization: ~4x less memory for vectors, minimal recall loss.
    Int8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadIndex {
    pub field: String,
    pub kind: PayloadIndexKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadIndexKind {
    Keyword,
    Integer,
    Float,
    Bool,
    Datetime,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingProvider {
    /// In-process ONNX models via fastembed (default, multilingual).
    FastEmbed,
    /// Any OpenAI-compatible `/embeddings` API (OpenAI, Azure, Ollama, vLLM, TEI, LiteLLM, ...).
    OpenAi,
    /// Voyage AI embeddings API.
    Voyage,
    /// Deterministic feature-hashing embedder. For tests and offline development only.
    Hash,
}

impl EmbeddingProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FastEmbed => "fastembed",
            Self::OpenAi => "openai",
            Self::Voyage => "voyage",
            Self::Hash => "hash",
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingSettings {
    pub provider: EmbeddingProvider,
    pub model: String,
    /// Output dimension. When unset it is discovered from the model. For OpenAI
    /// `text-embedding-3-*` and Voyage models it also requests a shortened vector.
    pub dimension: Option<usize>,
    /// Texts per embedding call.
    pub batch_size: usize,
    /// Maximum tokens per text (fastembed only).
    pub max_length: Option<usize>,
    /// Where downloaded models are cached (fastembed only).
    pub cache_dir: PathBuf,
    /// Number of model instances kept in memory (fastembed only).
    pub pool_size: usize,
    /// ONNX intra-op threads per model instance (fastembed only; default: all cores).
    pub threads: Option<usize>,
    /// Override the model's query prefix (`None` = use the model's recommended prefix).
    pub query_prefix: Option<String>,
    /// Override the model's document prefix (`None` = use the model's recommended prefix).
    pub document_prefix: Option<String>,
    pub api_base: Option<String>,
    pub api_key: Option<String>,
    pub timeout: Duration,
    pub max_retries: u32,
}

#[derive(Debug, Clone)]
pub struct TrainingSettings {
    /// Rows embedded and upserted per batch while training.
    pub batch_size: usize,
    /// Maximum training jobs running at the same time (embedding is CPU-heavy).
    pub max_concurrent_jobs: usize,
    /// Most recent items remembered per user (used to exclude already-seen items).
    pub profile_max_items: usize,
    /// A `training` dataset without progress for this long is considered interrupted.
    pub stale_job_after: Duration,
}

#[derive(Debug, Clone)]
pub struct RecommendSettings {
    pub default_limit: usize,
    pub max_limit: usize,
    pub max_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Pretty,
    Json,
}

impl Settings {
    /// Load settings from the process environment.
    pub fn from_env() -> anyhow::Result<Self> {
        Self::from_lookup(|key| std::env::var(key).ok())
    }

    /// Load settings from an arbitrary key/value lookup (used by tests).
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> anyhow::Result<Self> {
        let env = Env(lookup);

        let data_dir = PathBuf::from(env.string("DATA_DIR", "./data"));

        let server = ServerSettings {
            host: env.string("HOST", "0.0.0.0"),
            port: env.parse("PORT", 8000u16)?,
            workers: env.opt_parse("WORKERS")?,
            api_key: env.opt_string("API_KEY"),
            cors_allowed_origins: env
                .list("CORS_ALLOWED_ORIGINS")
                .unwrap_or_else(|| vec!["*".into()]),
            max_upload_bytes: env.parse::<usize>("MAX_UPLOAD_MB", 512)? * 1024 * 1024,
            json_limit_bytes: env.parse::<usize>("JSON_LIMIT_MB", 32)? * 1024 * 1024,
            swagger_ui: env.bool("SWAGGER_UI_ENABLED", true)?,
            metrics: env.bool("METRICS_ENABLED", true)?,
            shutdown_timeout_secs: env.parse("SHUTDOWN_TIMEOUT_SECS", 30u64)?,
        };

        let vector_store = VectorStoreSettings {
            kind: match env
                .string("VECTOR_STORE", "qdrant")
                .to_ascii_lowercase()
                .as_str()
            {
                "qdrant" => VectorStoreKind::Qdrant,
                "memory" | "in-memory" | "inmemory" => VectorStoreKind::Memory,
                other => bail!("invalid VECTOR_STORE '{other}' (expected 'qdrant' or 'memory')"),
            },
            qdrant: QdrantSettings {
                url: qdrant_url(&env),
                api_key: env.opt_string("QDRANT_API_KEY"),
                collection_prefix: env.string("QDRANT_COLLECTION_PREFIX", "recom"),
                timeout: Duration::from_secs(env.parse("QDRANT_TIMEOUT_SECS", 30u64)?),
                startup_timeout: Duration::from_secs(
                    env.parse("QDRANT_STARTUP_TIMEOUT_SECS", 60u64)?,
                ),
                on_disk_payload: env.bool("QDRANT_ON_DISK_PAYLOAD", true)?,
                quantization: match env
                    .string("QDRANT_QUANTIZATION", "none")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "none" | "off" | "" => Quantization::None,
                    "int8" | "scalar" => Quantization::Int8,
                    other => {
                        bail!("invalid QDRANT_QUANTIZATION '{other}' (expected 'none' or 'int8')")
                    }
                },
                shard_number: env.opt_parse("QDRANT_SHARD_NUMBER")?,
                replication_factor: env.opt_parse("QDRANT_REPLICATION_FACTOR")?,
                payload_indexes: parse_payload_indexes(
                    env.opt_string("QDRANT_PAYLOAD_INDEXES").as_deref(),
                )?,
            },
        };

        let provider = match env
            .opt_string("EMBEDDING_PROVIDER")
            .unwrap_or_else(|| "fastembed".into())
            .to_ascii_lowercase()
            .as_str()
        {
            "fastembed" | "local" => EmbeddingProvider::FastEmbed,
            "openai" | "openai-compatible" => EmbeddingProvider::OpenAi,
            "voyage" | "voyageai" => EmbeddingProvider::Voyage,
            "hash" => EmbeddingProvider::Hash,
            other => bail!(
                "invalid EMBEDDING_PROVIDER '{other}' (expected fastembed, openai, voyage or hash)"
            ),
        };

        let (default_model, default_batch) = match provider {
            EmbeddingProvider::FastEmbed => (DEFAULT_FASTEMBED_MODEL.to_string(), 32),
            EmbeddingProvider::OpenAi => (DEFAULT_OPENAI_MODEL.to_string(), 128),
            EmbeddingProvider::Voyage => (DEFAULT_VOYAGE_MODEL.to_string(), 128),
            EmbeddingProvider::Hash => (format!("hash-{DEFAULT_HASH_DIMENSION}"), 256),
        };

        let provider_key = match provider {
            EmbeddingProvider::OpenAi => env.opt_string("OPENAI_API_KEY"),
            EmbeddingProvider::Voyage => env.opt_string("VOYAGE_API_KEY"),
            _ => None,
        };

        let embedding = EmbeddingSettings {
            provider,
            model: env.opt_string("EMBEDDING_MODEL").unwrap_or(default_model),
            dimension: env.opt_parse("EMBEDDING_DIMENSION")?,
            batch_size: env.parse("EMBEDDING_BATCH_SIZE", default_batch)?,
            max_length: env.opt_parse("EMBEDDING_MAX_LENGTH")?,
            cache_dir: env
                .opt_string("EMBEDDING_CACHE_DIR")
                .or_else(|| env.opt_string("FASTEMBED_CACHE_DIR"))
                .map(PathBuf::from)
                .unwrap_or_else(|| data_dir.join("models")),
            pool_size: env.parse("EMBEDDING_POOL_SIZE", 1usize)?,
            threads: env.opt_parse("EMBEDDING_THREADS")?,
            query_prefix: env.raw("EMBEDDING_QUERY_PREFIX"),
            document_prefix: env.raw("EMBEDDING_DOCUMENT_PREFIX"),
            api_base: env.opt_string("EMBEDDING_API_BASE"),
            api_key: env.opt_string("EMBEDDING_API_KEY").or(provider_key),
            timeout: Duration::from_secs(env.parse("EMBEDDING_TIMEOUT_SECS", 60u64)?),
            max_retries: env.parse("EMBEDDING_MAX_RETRIES", 3u32)?,
        };

        let training = TrainingSettings {
            batch_size: env.parse("TRAINING_BATCH_SIZE", 256usize)?,
            max_concurrent_jobs: env.parse("TRAINING_MAX_CONCURRENT_JOBS", 2usize)?,
            profile_max_items: env.parse("USER_PROFILE_MAX_ITEMS", 500usize)?,
            stale_job_after: Duration::from_secs(env.parse("TRAINING_STALE_AFTER_SECS", 600u64)?),
        };

        let recommend = RecommendSettings {
            default_limit: env.parse("RECOMMEND_DEFAULT_LIMIT", 10usize)?,
            max_limit: env.parse("RECOMMEND_MAX_LIMIT", 200usize)?,
            max_offset: env.parse("RECOMMEND_MAX_OFFSET", 10_000usize)?,
        };

        let log_format = match env
            .string("LOG_FORMAT", "pretty")
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => LogFormat::Json,
            "pretty" | "text" | "plain" => LogFormat::Pretty,
            other => bail!("invalid LOG_FORMAT '{other}' (expected 'pretty' or 'json')"),
        };

        let settings = Settings {
            server,
            data_dir,
            vector_store,
            embedding,
            training,
            recommend,
            log_format,
            prediction_log_enabled: env.bool("PREDICTION_LOG_ENABLED", true)?,
        };
        settings.validate()?;
        Ok(settings)
    }

    /// Settings for hermetic tests and offline development: in-memory vector
    /// store, hash embedder and the given data directory.
    pub fn for_testing(data_dir: impl Into<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        let mut settings = Self::from_lookup(|key| match key {
            "VECTOR_STORE" => Some("memory".into()),
            "EMBEDDING_PROVIDER" => Some("hash".into()),
            "LOG_FORMAT" => Some("pretty".into()),
            _ => None,
        })
        .expect("default test settings are valid");
        settings.embedding.cache_dir = data_dir.join("models");
        settings.data_dir = data_dir;
        settings
    }

    fn validate(&self) -> anyhow::Result<()> {
        if self.embedding.batch_size == 0 {
            bail!("EMBEDDING_BATCH_SIZE must be greater than 0");
        }
        if self.embedding.pool_size == 0 {
            bail!("EMBEDDING_POOL_SIZE must be greater than 0");
        }
        if self.training.batch_size == 0 {
            bail!("TRAINING_BATCH_SIZE must be greater than 0");
        }
        if self.training.max_concurrent_jobs == 0 {
            bail!("TRAINING_MAX_CONCURRENT_JOBS must be greater than 0");
        }
        if self.recommend.default_limit == 0
            || self.recommend.default_limit > self.recommend.max_limit
        {
            bail!("RECOMMEND_DEFAULT_LIMIT must be between 1 and RECOMMEND_MAX_LIMIT");
        }
        if self.embedding.dimension == Some(0) {
            bail!("EMBEDDING_DIMENSION must be greater than 0");
        }
        if self.embedding.provider == EmbeddingProvider::FastEmbed && !cfg!(feature = "fastembed") {
            bail!(
                "EMBEDDING_PROVIDER=fastembed requires the `fastembed` cargo feature; \
                 rebuild with default features or choose another provider"
            );
        }
        if self.embedding.provider == EmbeddingProvider::Voyage && self.embedding.api_key.is_none()
        {
            bail!("EMBEDDING_PROVIDER=voyage requires EMBEDDING_API_KEY (or VOYAGE_API_KEY)");
        }
        let prefix = &self.vector_store.qdrant.collection_prefix;
        if prefix.is_empty()
            || !prefix
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            bail!("QDRANT_COLLECTION_PREFIX must be non-empty and contain only [A-Za-z0-9_-]");
        }
        Ok(())
    }
}

/// `QDRANT_URL` wins; otherwise the legacy `QDRANT_HOST`/`QDRANT_PORT`/`QDRANT_HTTPS` trio is used.
fn qdrant_url<F: Fn(&str) -> Option<String>>(env: &Env<F>) -> String {
    if let Some(url) = env.opt_string("QDRANT_URL") {
        return url;
    }
    let host = env.string("QDRANT_HOST", "localhost");
    let port = env.string("QDRANT_PORT", "6334");
    let https = env.bool("QDRANT_HTTPS", false).unwrap_or(false);
    format!("{}://{host}:{port}", if https { "https" } else { "http" })
}

/// Parses `field:type,field:type` (e.g. `category:keyword,price:float`).
fn parse_payload_indexes(raw: Option<&str>) -> anyhow::Result<Vec<PayloadIndex>> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|spec| {
            let (field, kind) = spec.split_once(':').ok_or_else(|| {
                anyhow!("invalid QDRANT_PAYLOAD_INDEXES entry '{spec}' (expected field:type)")
            })?;
            let kind = match kind.trim().to_ascii_lowercase().as_str() {
                "keyword" | "string" => PayloadIndexKind::Keyword,
                "integer" | "int" => PayloadIndexKind::Integer,
                "float" | "number" => PayloadIndexKind::Float,
                "bool" | "boolean" => PayloadIndexKind::Bool,
                "datetime" => PayloadIndexKind::Datetime,
                "text" => PayloadIndexKind::Text,
                other => bail!("invalid payload index type '{other}' for field '{field}'"),
            };
            Ok(PayloadIndex {
                field: field.trim().to_string(),
                kind,
            })
        })
        .collect()
}

struct Env<F>(F);

impl<F: Fn(&str) -> Option<String>> Env<F> {
    /// Raw value, including empty strings (meaningful for prefixes).
    fn raw(&self, key: &str) -> Option<String> {
        (self.0)(key)
    }

    /// Trimmed, non-empty value.
    fn opt_string(&self, key: &str) -> Option<String> {
        self.raw(key)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    }

    fn string(&self, key: &str, default: &str) -> String {
        self.opt_string(key).unwrap_or_else(|| default.to_string())
    }

    fn opt_parse<T>(&self, key: &str) -> anyhow::Result<Option<T>>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.opt_string(key)
            .map(|v| {
                v.parse::<T>()
                    .map_err(|e| anyhow!("{e}"))
                    .with_context(|| format!("invalid value for {key}: '{v}'"))
            })
            .transpose()
    }

    fn parse<T>(&self, key: &str, default: T) -> anyhow::Result<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        Ok(self.opt_parse(key)?.unwrap_or(default))
    }

    fn bool(&self, key: &str, default: bool) -> anyhow::Result<bool> {
        match self.opt_string(key).map(|v| v.to_ascii_lowercase()) {
            None => Ok(default),
            Some(v) => match v.as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => bail!("invalid boolean for {key}: '{v}'"),
            },
        }
    }

    fn list(&self, key: &str) -> Option<Vec<String>> {
        self.opt_string(key).map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn load(pairs: &[(&str, &str)]) -> anyhow::Result<Settings> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Settings::from_lookup(move |k| map.get(k).cloned())
    }

    #[test]
    fn defaults_are_sensible() {
        let s = load(&[]).unwrap();
        assert_eq!(s.server.port, 8000);
        assert_eq!(s.vector_store.kind, VectorStoreKind::Qdrant);
        assert_eq!(s.vector_store.qdrant.url, "http://localhost:6334");
        assert_eq!(s.embedding.provider, EmbeddingProvider::FastEmbed);
        assert_eq!(s.embedding.model, DEFAULT_FASTEMBED_MODEL);
        assert_eq!(s.embedding.cache_dir, PathBuf::from("./data/models"));
        assert_eq!(s.server.cors_allowed_origins, vec!["*"]);
        assert!(s.embedding.query_prefix.is_none());
    }

    #[test]
    fn legacy_qdrant_host_port_are_supported() {
        let s = load(&[
            ("QDRANT_HOST", "qdrant"),
            ("QDRANT_PORT", "6336"),
            ("QDRANT_HTTPS", "True"),
        ])
        .unwrap();
        assert_eq!(s.vector_store.qdrant.url, "https://qdrant:6336");
        let s = load(&[("QDRANT_URL", "http://q:6334"), ("QDRANT_HOST", "ignored")]).unwrap();
        assert_eq!(s.vector_store.qdrant.url, "http://q:6334");
    }

    #[test]
    fn provider_specific_defaults() {
        let s = load(&[
            ("EMBEDDING_PROVIDER", "openai"),
            ("OPENAI_API_KEY", "sk-test"),
        ])
        .unwrap();
        assert_eq!(s.embedding.model, DEFAULT_OPENAI_MODEL);
        assert_eq!(s.embedding.api_key.as_deref(), Some("sk-test"));

        assert!(
            load(&[("EMBEDDING_PROVIDER", "voyage")]).is_err(),
            "voyage requires a key"
        );
        let s = load(&[("EMBEDDING_PROVIDER", "voyage"), ("VOYAGE_API_KEY", "k")]).unwrap();
        assert_eq!(s.embedding.model, DEFAULT_VOYAGE_MODEL);
    }

    #[test]
    fn empty_prefix_is_an_explicit_override() {
        let s = load(&[
            ("EMBEDDING_QUERY_PREFIX", ""),
            ("EMBEDDING_DOCUMENT_PREFIX", "doc: "),
        ])
        .unwrap();
        assert_eq!(s.embedding.query_prefix.as_deref(), Some(""));
        assert_eq!(s.embedding.document_prefix.as_deref(), Some("doc: "));
    }

    #[test]
    fn parses_payload_indexes() {
        let s = load(&[(
            "QDRANT_PAYLOAD_INDEXES",
            "category:keyword, price:float,in_stock:bool",
        )])
        .unwrap();
        assert_eq!(
            s.vector_store.qdrant.payload_indexes,
            vec![
                PayloadIndex {
                    field: "category".into(),
                    kind: PayloadIndexKind::Keyword
                },
                PayloadIndex {
                    field: "price".into(),
                    kind: PayloadIndexKind::Float
                },
                PayloadIndex {
                    field: "in_stock".into(),
                    kind: PayloadIndexKind::Bool
                },
            ]
        );
        assert!(load(&[("QDRANT_PAYLOAD_INDEXES", "category")]).is_err());
        assert!(load(&[("QDRANT_PAYLOAD_INDEXES", "category:blob")]).is_err());
    }

    #[test]
    fn rejects_invalid_values() {
        assert!(load(&[("PORT", "eighty")]).is_err());
        assert!(load(&[("VECTOR_STORE", "faiss")]).is_err());
        assert!(load(&[("METRICS_ENABLED", "maybe")]).is_err());
        assert!(load(&[("TRAINING_BATCH_SIZE", "0")]).is_err());
        assert!(load(&[("QDRANT_COLLECTION_PREFIX", "bad prefix")]).is_err());
    }

    #[test]
    fn testing_settings_use_memory_and_hash() {
        let s = Settings::for_testing("/tmp/x");
        assert_eq!(s.vector_store.kind, VectorStoreKind::Memory);
        assert_eq!(s.embedding.provider, EmbeddingProvider::Hash);
        assert_eq!(s.data_dir, PathBuf::from("/tmp/x"));
    }
}
