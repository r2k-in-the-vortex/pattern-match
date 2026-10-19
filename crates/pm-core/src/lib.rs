//! `pm-core` — pattern-match algorithms and data types.
//!
//! The crate is independent of the UI and file dialogs. The module layout
//! follows the architecture in the project plan (README §5.2); implementations
//! arrive in milestones M1–M6. Modules that are still empty carry a doc comment
//! naming their responsibility and milestone.

pub mod capture;
pub mod corner;
pub mod descriptor;
pub mod edge;
pub mod error;
pub mod feature;
pub mod geometry;
pub mod gradient;
pub mod gray;
pub mod model;
pub mod overlay;
pub mod pyramid;
pub mod refine;
pub mod score;
pub mod search;
pub mod segment;

pub use error::{Error, Result};
pub use feature::{
    Corner, EdgeChain, EdgePoint, EdgeSet, FeatureParams, FeatureSet, Segment, extract_features,
};
pub use gray::{GrayF32, LoadedImage, load_image};
pub use image::RgbaImage;
pub use overlay::{OverlayOptions, render_overlay};
pub use pyramid::Pyramid;

/// Version of the `pm-core` crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
