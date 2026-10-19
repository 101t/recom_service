//! Business logic, independent of HTTP.
//!
//! * [`DatasetService`] — datasets, item indexing ("train"), interactions and user profiles.
//! * [`Recommender`] — serving recommendations ("predict").
//! * [`PredictionLogger`] — asynchronous prediction logs and exports.

mod datasets;
mod indexing;
pub mod prediction_log;
mod profiles;
mod recommender;

pub use datasets::{Accepted, DatasetService, Upload};
pub use indexing::MAX_ITEMS_PER_REQUEST;
pub use prediction_log::{Export, PredictionLogger};
pub use profiles::MAX_INTERACTIONS_PER_REQUEST;
pub use recommender::Recommender;
