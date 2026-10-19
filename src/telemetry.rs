//! Logging setup. Verbosity follows `RUST_LOG`; `LOG_FORMAT=json` emits one JSON
//! object per line for log aggregators.

use tracing_subscriber::EnvFilter;

use crate::config::LogFormat;

const DEFAULT_FILTER: &str = "info,ort=warn,h2=warn,hyper=warn,tower=warn,hf_hub=warn";

/// Installs the global subscriber. Safe to call more than once.
pub fn init(format: LogFormat) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    let _ = match format {
        LogFormat::Json => builder
            .json()
            .flatten_event(true)
            .with_current_span(false)
            .try_init(),
        LogFormat::Pretty => builder.compact().try_init(),
    };
}
