//! Persistence of dataset metadata and raw data (the "source of truth" a
//! dataset's index can always be rebuilt from).

pub mod datasets;
pub mod tabular;

pub use datasets::{DatasetRepository, ItemPart};
