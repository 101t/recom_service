//! `recommendation-service` binary.
//!
//! ```text
//! recommendation-service [serve]        start the HTTP server (default)
//! recommendation-service healthcheck    exit 0 if the local server answers /health (for Docker)
//! recommendation-service download-model download the configured embedding model and exit
//! recommendation-service version        print the version
//! ```

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::process::ExitCode;
use std::time::Duration;

use recom_service::config::Settings;
use recom_service::{embedding, server, telemetry};

const USAGE: &str = "usage: recommendation-service [serve|healthcheck|download-model|version|help]";

fn main() -> ExitCode {
    let command = std::env::args().nth(1).unwrap_or_else(|| "serve".into());
    match command.as_str() {
        "serve" => run_async(serve()),
        "download-model" => run_async(download_model()),
        "healthcheck" => healthcheck(),
        "version" | "--version" | "-V" => {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        "help" | "--help" | "-h" => {
            println!(
                "{USAGE}\n\nConfiguration is read from environment variables (and .env); see .env.example."
            );
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command '{other}'\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn load_settings() -> anyhow::Result<Settings> {
    dotenvy::dotenv().ok();
    let settings = Settings::from_env()?;
    telemetry::init(settings.log_format);
    Ok(settings)
}

fn run_async(future: impl Future<Output = anyhow::Result<()>>) -> ExitCode {
    // HTTP workers get their own single-threaded runtimes from actix. The main
    // runtime is multi-threaded because it drives the Qdrant connection,
    // prediction logging and background training jobs.
    let system = actix_web::rt::System::with_tokio_rt(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("recom-main")
            .build()
            .expect("failed to build the Tokio runtime")
    });
    match system.block_on(future) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            tracing::error!(error = format!("{err:#}"), "fatal error");
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

async fn serve() -> anyhow::Result<()> {
    server::run(load_settings()?).await
}

async fn download_model() -> anyhow::Result<()> {
    let settings = load_settings()?;
    let embedder = embedding::build(&settings.embedding).await?;
    println!(
        "{} ({} dimensions) is ready",
        embedder.model(),
        embedder.dimension()
    );
    Ok(())
}

/// Minimal HTTP probe so container images need no curl/wget.
fn healthcheck() -> ExitCode {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let probe = || -> std::io::Result<bool> {
        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
        stream.set_read_timeout(Some(Duration::from_secs(3)))?;
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
        let mut buf = [0u8; 32];
        let n = stream.read(&mut buf)?;
        Ok(buf[..n].starts_with(b"HTTP/1.1 200"))
    };
    match probe() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("unhealthy: /health did not return 200");
            ExitCode::FAILURE
        }
        Err(err) => {
            eprintln!("unhealthy: {err}");
            ExitCode::FAILURE
        }
    }
}
