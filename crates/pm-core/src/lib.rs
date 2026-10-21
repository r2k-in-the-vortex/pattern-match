//! `pm-core` — pattern-match algorithms and data types.
//!
//! The crate is independent of the UI and file dialogs. The module layout
//! follows the architecture in the project plan (README §5.2). Detection and
//! model functionality through M6 is implemented; `capture` remains a stub
//! until the webcam milestone.

pub mod capture;
pub mod corner;
pub mod descriptor;
pub mod edge;
pub mod error;
pub mod feature;
pub mod geometry;
pub mod gradient;
pub mod gray;
pub mod mask;
pub mod model;
pub mod overlay;
pub mod pyramid;
pub mod refine;
pub mod result;
pub mod score;
pub mod search;
pub mod segment;

pub use descriptor::{
    CornerDescriptor, SegmentPairInvariant, describe_corners, segment_pair_invariants,
};
pub use error::{Error, Result};
pub use feature::{
    Corner, EdgeChain, EdgePoint, EdgeSet, FeatureParams, FeatureSet, Segment, extract_features,
    filter_features,
};
pub use geometry::{
    Homography, PoseEstimate, RotatedRect, Similarity, ZeroFrame, has_degenerate_triple,
    normalized_dlt,
};
pub use gray::{GrayF32, LoadedImage, load_image};
pub use image::RgbaImage;
pub use mask::{MaskBounds, MaskPrimitive, MaskRaster};
pub use model::{HypothesisSources, Model, SearchDefaults, TeachSettings, mask_bounds_for};
pub use overlay::{OverlayOptions, render_overlay};
pub use pyramid::Pyramid;
pub use refine::{
    Correspondence, HomographyFallbackReason, HomographyInput, HomographyOutcome, HomographyParams,
    RansacParams, RansacResult, SimilarityRefineParams, ransac_homography,
    ransac_homography_correspondences, refine_homography, refine_similarity_level,
    required_iterations, required_iterations_from_q, validate_homography,
};
pub use result::{
    DetectionRecord, ImageInfo, PoseRecord, RESULTS_FORMAT, RESULTS_VERSION, ResultsFile,
    TimingsRecord,
};
pub use score::{
    Detection, DetectionDiagnostics, NmsParams, ScoreParams, SeedSource, confidence_from,
    non_maximum_suppression, score_homography, score_similarity,
};
pub use search::{
    DetectionResult, Hypothesis, SearchParams, SearchTimings, TestLevel, TestPyramid, detect,
    hough_hypotheses,
};

/// Version of the `pm-core` crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
