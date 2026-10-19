//! Recommendation service library.
//!
//! The binary (`src/main.rs`) is a thin wrapper around [`server::run`]; everything
//! else lives here so it can be reused and tested.
//!
//! Layers, from the outside in:
//! * [`api`] — HTTP handlers and the OpenAPI document;
//! * [`services`] — datasets/training, recommendations, prediction logs;
//! * [`embedding`] and [`vector_store`] — pluggable providers behind traits;
//! * [`storage`] — dataset metadata and raw data on disk.

pub mod api;
pub mod config;
pub mod embedding;
pub mod error;
pub mod ids;
pub mod models;
pub mod server;
pub mod services;
pub mod state;
pub mod storage;
pub mod telemetry;
pub mod vector_store;
