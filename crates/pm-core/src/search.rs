//! Hybrid coarse similarity search: oriented Hough voting, corner-descriptor
//! and segment-pair seeding, hypothesis clustering, coarse-to-fine
//! verification, refinement, and multi-instance NMS (README §9.1–§9.8).
//!
//! The similarity search is the M5 detector.  After full-resolution refinement
//! each candidate passes through the M6 homography stage ([`refine_homography`]):
//! correspondences are assembled, RANSAC and robust LM estimate an 8-DOF
//! homography, and a validation gate either accepts it or reports the
//! similarity embedding as the auditable fallback.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nalgebra::Vector2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::corner::detect_corners;
use crate::descriptor::{
    CornerDescriptor, DESCRIPTOR_RADII, DESCRIPTOR_SCALE_DIM, describe_corners,
    segment_pair_invariants,
};
use crate::edge::{DistanceField, extract_edges};
use crate::error::{Error, Result};
use crate::feature::{Corner, EdgePoint, EdgeSet, FeatureParams, Segment};
use crate::geometry::{RotatedRect, Similarity, matrix_to_array};
use crate::gradient::GradientField;
use crate::model::{HypothesisSources, Model, SearchDefaults};
use crate::pyramid::Pyramid;
use crate::refine::{
    HomographyInput, HomographyParams, SimilarityRefineParams, refine_homography,
    refine_similarity_level,
};
use crate::score::{
    Detection, DetectionDiagnostics, NmsParams, ScoreDiagnostics, ScoreParams, SeedSource,
    confidence_from, non_maximum_suppression, score_similarity,
};
use crate::segment::fit_segments;

/// Number of directed orientation buckets in Hough voting.
pub const HOUGH_ORIENTATION_BUCKETS: usize = 32;

/// Stage of the detection pipeline reported to a [`ProgressSink`]
/// (README §11.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectStage {
    /// Blurred base image and Gaussian pyramid construction.
    Preprocess,
    /// Gradients, edges, distance fields, corners, and segments.
    Edges,
    /// Hough voting plus descriptor and segment seeding.
    Search,
    /// Hypothesis verification and refinement.
    Verify,
    /// Similarity refinement and non-maximum suppression.
    Refine,
    /// Homography correspondence assembly, RANSAC, and LM refinement.
    Homography,
}

impl DetectStage {
    /// Stable display name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Preprocess => "preprocess",
            Self::Edges => "edges",
            Self::Search => "Hough/descriptor seeding",
            Self::Verify => "verify",
            Self::Refine => "refine",
            Self::Homography => "homography",
        }
    }
}

/// Optional cooperative progress and cancellation hook for
/// [`detect_with_progress`].
///
/// `progress` reports a stage and a whole-run fraction in `[0, 1]`.
/// `cancelled` is polled between work units; returning `true` makes the
/// detector return [`Error::Cancelled`] without producing a result.  The
/// default implementations do nothing, so [`IgnoreProgress`] observes nothing
/// and never cancels.
pub trait ProgressSink: Send + Sync {
    /// Report the active stage and whole-run fraction in `[0, 1]`.
    fn progress(&self, _stage: DetectStage, _fraction: f32) {}

    /// Whether the caller asked to abort the run.
    fn cancelled(&self) -> bool {
        false
    }
}

/// A [`ProgressSink`] that reports nothing and never cancels.
#[derive(Clone, Copy, Debug, Default)]
pub struct IgnoreProgress;

impl ProgressSink for IgnoreProgress {}

/// A test image level with gradients, edges, distance field, and features.
#[derive(Clone, Debug)]
pub struct TestLevel {
    /// Pyramid level index.
    pub level: usize,
    /// Grayscale image at this level.
    pub gray: crate::gray::GrayF32,
    /// Gradient field.
    pub gradient: GradientField,
    /// Accepted edge points.
    pub edges: EdgeSet,
    /// Distance to the nearest edge pixel.
    pub distance: DistanceField,
    /// Shi–Tomasi corners.
    pub corners: Vec<Corner>,
    /// Fitted segments.
    pub segments: Vec<Segment>,
}

/// A preprocessed test-image pyramid.
#[derive(Clone, Debug)]
pub struct TestPyramid {
    /// Levels from full resolution upward.
    pub levels: Vec<TestLevel>,
    /// Time to build the blurred base image and the Gaussian pyramid.
    pub preprocess_ms: f64,
    /// Time to compute gradients, edges, distance fields, and features.
    pub features_ms: f64,
}

impl TestPyramid {
    /// Build the test pyramid and per-level features.
    ///
    /// # Panics
    ///
    /// Never in practice: this uses [`IgnoreProgress`], which cannot cancel.
    #[must_use]
    pub fn build(image: &crate::gray::GrayF32, levels: usize, params: &FeatureParams) -> Self {
        Self::build_with_progress(image, levels, params, &IgnoreProgress)
            .expect("IgnoreProgress never cancels")
    }

    /// Build the test pyramid, reporting progress and honouring cancellation
    /// between pyramid levels.
    pub fn build_with_progress(
        image: &crate::gray::GrayF32,
        levels: usize,
        params: &FeatureParams,
        progress: &dyn ProgressSink,
    ) -> Result<Self> {
        progress.progress(DetectStage::Preprocess, 0.0);
        let preprocess_start = Instant::now();
        let base = if params.blur_sigma > 0.01 {
            image.gaussian_blur(params.blur_sigma)
        } else {
            image.clone()
        };
        let pyramid = Pyramid::build(&base, levels);
        let preprocess_ms = duration_ms(preprocess_start);
        if progress.cancelled() {
            return Err(Error::Cancelled);
        }

        let features_start = Instant::now();
        let level_count = pyramid.len().max(1);
        let built: Vec<Option<TestLevel>> = (0..pyramid.len())
            .into_par_iter()
            .map(|level| {
                if progress.cancelled() {
                    return None;
                }
                let gray = pyramid.level_unchecked(level).clone();
                let gradient = GradientField::compute(&gray);
                let edges = extract_edges(&gradient, &params.edge);
                let distance = DistanceField::from_edges(&edges, gray.width, gray.height);
                let corners = if params.detect_corners {
                    detect_corners(&gradient, &params.corner)
                } else {
                    Vec::new()
                };
                let segments = if params.fit_segments {
                    fit_segments(&edges, &params.segment)
                } else {
                    Vec::new()
                };
                progress.progress(
                    DetectStage::Edges,
                    0.02 + 0.28 * (level + 1) as f32 / level_count as f32,
                );
                Some(TestLevel {
                    level,
                    gray,
                    gradient,
                    edges,
                    distance,
                    corners,
                    segments,
                })
            })
            .collect();
        if built.iter().any(Option::is_none) {
            return Err(Error::Cancelled);
        }
        let built: Vec<TestLevel> = built.into_iter().flatten().collect();
        Ok(Self {
            levels: built,
            preprocess_ms,
            features_ms: duration_ms(features_start),
        })
    }

    /// A level, or `None` when out of range.
    #[must_use]
    pub fn level(&self, level: usize) -> Option<&TestLevel> {
        self.levels.get(level)
    }
}

/// Search parameters for [`detect`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchParams {
    /// Minimum rotation in degrees.
    pub rotation_min_deg: f32,
    /// Maximum rotation in degrees.
    pub rotation_max_deg: f32,
    /// Minimum scale.
    pub scale_min: f32,
    /// Maximum scale.
    pub scale_max: f32,
    /// Rotation bins at the coarse level.
    pub rotation_bins: usize,
    /// Log-spaced scale bins at the coarse level.
    pub scale_bins: usize,
    /// Smooth orientation gate in radians.
    pub orientation_gate: f32,
    /// Fraction of the strongest peak kept in each bin.
    pub peak_fraction: f32,
    /// Peaks per `(rotation, scale)` bin.
    pub peaks_per_bin: usize,
    /// Descriptor ratio-test threshold.
    pub descriptor_ratio: f32,
    /// Maximum model corners used for descriptor matching.
    pub model_corner_cap: usize,
    /// Maximum test corners per level used for descriptor matching.
    pub test_corner_cap: usize,
    /// Maximum segment pairs per image.
    pub segment_pair_cap: usize,
    /// Maximum clustered hypotheses kept before coarse ranking.
    pub hypothesis_cap: usize,
    /// Maximum coarse-ranked hypotheses submitted to full verification.
    pub max_hypotheses_scored: usize,
    /// Maximum Hough model edge points after deterministic subsampling.
    pub max_model_hough_points: usize,
    /// Maximum Hough test edge points after deterministic subsampling.
    pub max_test_hough_points: usize,
    /// Optional manual coarse level.
    pub coarse_level: Option<usize>,
    /// Optional search region in whole-image coordinates.
    pub search_region: Option<RotatedRect>,
    /// Hypothesis sources to enable.
    pub hypothesis_sources: HypothesisSources,
    /// Minimum accepted score.
    pub score_threshold: f32,
    /// Minimum visible weight.
    pub coverage_min: f32,
    /// Whether polarity must agree.
    pub polarity_enabled: bool,
    /// Maximum detections.
    pub max_detections: usize,
    /// Scoring parameters.
    pub score: ScoreParams,
    /// Similarity refinement parameters.
    pub refine: SimilarityRefineParams,
    /// Homography estimation and fallback parameters.
    pub homography: HomographyParams,
    /// NMS thresholds.
    pub nms: NmsParams,
}

impl SearchParams {
    /// Default Hough orientation gate in radians (speed-oriented).
    ///
    /// The gate window `2 * gate` covers the rotation step so every
    /// orientation still passes at least one bin while each pair votes in
    /// fewer bins.
    pub const FAST_ORIENTATION_GATE: f32 = 0.08;

    /// Hough orientation gate used by the high-accuracy mode in radians.
    ///
    /// This is the pre-optimization value: a wider kernel lets noisy edges
    /// contribute to more bins, which slightly improves recall on
    /// low-contrast and cluttered scenes at the cost of Hough time.
    pub const HIGH_ACCURACY_ORIENTATION_GATE: f32 = 0.20;
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            rotation_min_deg: -180.0,
            rotation_max_deg: 180.0,
            scale_min: 0.5,
            scale_max: 2.0,
            // The gate must satisfy 2 * gate >= step so every orientation
            // passes at least one rotation bin: 2 * 0.08 = 0.16 rad covers the
            // 0.106 rad step of 60 bins.  The tighter gate is what makes each
            // test/model pair vote in fewer bins; coarser rotation or scale
            // bins move poses far enough to trip the homography fallback gate.
            rotation_bins: 60,
            scale_bins: 12,
            orientation_gate: Self::FAST_ORIENTATION_GATE,
            peak_fraction: 0.25,
            peaks_per_bin: 2,
            descriptor_ratio: 0.8,
            model_corner_cap: 300,
            test_corner_cap: 600,
            segment_pair_cap: 500,
            hypothesis_cap: 150,
            max_hypotheses_scored: 24,
            max_model_hough_points: 400,
            max_test_hough_points: 1500,
            coarse_level: None,
            search_region: None,
            hypothesis_sources: HypothesisSources::Both,
            score_threshold: 0.55,
            coverage_min: 0.4,
            polarity_enabled: true,
            max_detections: 10,
            score: ScoreParams::default(),
            refine: SimilarityRefineParams::default(),
            homography: HomographyParams::default(),
            nms: NmsParams::default(),
        }
    }
}

impl From<SearchDefaults> for SearchParams {
    fn from(defaults: SearchDefaults) -> Self {
        Self {
            rotation_min_deg: defaults.rotation_min_deg,
            rotation_max_deg: defaults.rotation_max_deg,
            scale_min: defaults.scale_min,
            scale_max: defaults.scale_max,
            score_threshold: defaults.score_threshold,
            coverage_min: defaults.coverage_min,
            polarity_enabled: defaults.polarity_enabled,
            hypothesis_sources: defaults.hypothesis_sources,
            max_detections: defaults.max_detections,
            score: ScoreParams {
                coverage_min: defaults.coverage_min,
                polarity_enabled: defaults.polarity_enabled,
                ..ScoreParams::default()
            },
            homography: HomographyParams {
                ransac: crate::refine::RansacParams {
                    confidence: f64::from(defaults.ransac_confidence),
                    seed: defaults.ransac_seed,
                    ..crate::refine::RansacParams::default()
                },
                ..HomographyParams::default()
            },
            ..Self::default()
        }
    }
}

/// A coarse similarity hypothesis in whole-image coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypothesis {
    /// Full-resolution pose.
    pub pose: Similarity,
    /// Source-specific seed score, used for ranking and clustering.
    pub seed_score: f32,
    /// Which coarse stage produced it.
    pub source: SeedSource,
    /// Level the seed was generated at.
    pub level: usize,
}

/// Stage timings in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchTimings {
    /// Blurred base image and Gaussian pyramid construction time.
    pub preprocess_ms: f64,
    /// Gradients, edges, distance fields, corners, and segments per level.
    pub features_ms: f64,
    /// Hough and descriptor/segment hypothesis time.
    pub search_ms: f64,
    /// Coarse-to-fine scoring and refinement time.
    pub refine_ms: f64,
    /// End-to-end time.
    pub total_ms: f64,
}

/// Detector output.
///
/// Use [`crate::result::ResultsFile`] to serialize this as the versioned JSON
/// result schema; the derived serde form here is internal.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Accepted detections after NMS and thresholding.
    pub detections: Vec<Detection>,
    /// Stage timings.
    pub timings: SearchTimings,
    /// Number of raw hypotheses before clustering.
    pub hypotheses_generated: usize,
    /// Number of hypotheses verified after clustering.
    pub hypotheses_scored: usize,
}

/// Detect instances of `model` in `image` (README §9.1).
///
/// This is the default entry point.  It performs no progress reporting and
/// cannot be cancelled; use [`detect_with_progress`] for a cooperative hook.
pub fn detect(
    model: &Model,
    image: &crate::gray::GrayF32,
    params: &SearchParams,
) -> Result<DetectionResult> {
    detect_with_progress(model, image, params, &IgnoreProgress)
}

/// Detect instances with cooperative progress and cancellation (README §11.3).
///
/// The hook never changes which detections are produced: with a sink that never
/// cancels, results are identical to [`detect`].  Cancellation is polled between
/// pyramid levels, hypothesis batches, ranking work, and verification
/// candidates; a cancelled run returns [`Error::Cancelled`] instead of a
/// partially computed result.
pub fn detect_with_progress(
    model: &Model,
    image: &crate::gray::GrayF32,
    params: &SearchParams,
    progress: &dyn ProgressSink,
) -> Result<DetectionResult> {
    let total_start = Instant::now();
    if image.is_empty() {
        return Err(Error::InvalidParameter("test image is empty".to_owned()));
    }
    if model.features.edge_count() == 0 {
        return Err(Error::InvalidParameter(
            "model has no edge features".to_owned(),
        ));
    }
    if progress.cancelled() {
        return Err(Error::Cancelled);
    }
    let mut timings = SearchTimings::default();

    let test = TestPyramid::build_with_progress(image, 3, &model.params, progress)?;
    timings.preprocess_ms = test.preprocess_ms;
    timings.features_ms = test.features_ms;
    if test.levels.is_empty() {
        timings.total_ms = duration_ms(total_start);
        return Ok(DetectionResult {
            timings,
            ..DetectionResult::default()
        });
    }

    let coarse_level = select_coarse_level(model, params, test.levels.len());
    let model_levels: Vec<Vec<EdgePoint>> = (0..=coarse_level)
        .map(|level| model.features_at_level(level).edges.points)
        .collect();

    let search_start = Instant::now();
    progress.progress(DetectStage::Search, 0.30);
    let mut hypotheses = Vec::new();
    if matches!(
        params.hypothesis_sources,
        HypothesisSources::Hough | HypothesisSources::Both
    ) {
        hypotheses.extend(hough_hypotheses_with_progress(
            model,
            &test.levels[coarse_level],
            coarse_level,
            params,
            progress,
        ));
    }
    if progress.cancelled() {
        return Err(Error::Cancelled);
    }
    if matches!(
        params.hypothesis_sources,
        HypothesisSources::Descriptors | HypothesisSources::Both
    ) {
        hypotheses.extend(descriptor_hypotheses_with_progress(
            model,
            &test.levels,
            coarse_level,
            params,
            progress,
        ));
        if progress.cancelled() {
            return Err(Error::Cancelled);
        }
        hypotheses.extend(segment_hypotheses_with_progress(
            model,
            &test.levels,
            coarse_level,
            params,
            progress,
        ));
    }
    if progress.cancelled() {
        return Err(Error::Cancelled);
    }
    let generated = hypotheses.len();
    let model_diagonal = model_roi_diagonal(model);
    let clustered = cluster_hypotheses(hypotheses, model_diagonal, params);
    let ranked = rank_hypotheses(
        clustered,
        &test.levels[coarse_level],
        &model_levels[coarse_level],
        coarse_level,
        model_diagonal,
        params,
        progress,
    );
    if progress.cancelled() {
        return Err(Error::Cancelled);
    }
    timings.search_ms = duration_ms(search_start);

    let refine_start = Instant::now();
    progress.progress(DetectStage::Verify, 0.55);
    let hypotheses_scored = ranked.len();
    let roi_model = model_roi_in_model_coords(model);
    let completed = AtomicUsize::new(0);
    let total = ranked.len().max(1);
    let scored: Vec<Option<Detection>> = ranked
        .par_iter()
        .enumerate()
        .map(|(index, hypothesis)| {
            verify_hypothesis(
                model,
                &test.levels,
                &model_levels,
                coarse_level,
                index,
                hypothesis,
                params,
                progress,
                &completed,
                total,
            )
        })
        .collect();
    if progress.cancelled() {
        return Err(Error::Cancelled);
    }
    timings.refine_ms = duration_ms(refine_start);

    let mut detections: Vec<Detection> = scored.into_iter().flatten().collect();
    detections.retain(|detection| {
        detection.score >= params.score_threshold
            && detection.diagnostics.coverage >= params.coverage_min
            && params.search_region.is_none_or(|region| {
                region.contains(detection.pose.tx as f32, detection.pose.ty as f32)
            })
    });
    let detections = non_maximum_suppression(
        detections,
        model_diagonal,
        &roi_model,
        params.max_detections,
        &params.nms,
    );
    progress.progress(DetectStage::Refine, 1.0);
    timings.total_ms = duration_ms(total_start);
    Ok(DetectionResult {
        detections,
        timings,
        hypotheses_generated: generated,
        hypotheses_scored,
    })
}

fn duration_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Model ROI diagonal in taught-image pixels.
#[must_use]
pub fn model_roi_diagonal(model: &Model) -> f32 {
    model.roi.width.hypot(model.roi.height)
}

/// ROI corners converted into model coordinates.
#[must_use]
pub fn model_roi_in_model_coords(model: &Model) -> [[f32; 2]; 4] {
    model
        .roi
        .corners()
        .map(|corner| model.zero.to_model(corner))
}

/// Choose the coarsest level whose ROI short side is at least 32 level pixels
/// and whose model still has at least 120 edge points (README §9.2).
#[must_use]
pub fn select_coarse_level(model: &Model, params: &SearchParams, available: usize) -> usize {
    if available == 0 {
        return 0;
    }
    if let Some(level) = params.coarse_level {
        return level.min(available - 1);
    }
    let short = model.roi.width.min(model.roi.height);
    let points = model.features.edge_count();
    let mut chosen = 0usize;
    for level in 0..available {
        let scale = (1u32 << level) as f32;
        if short / scale >= 32.0 && points >= 120 {
            chosen = level;
        } else {
            break;
        }
    }
    chosen
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn angle_distance(a: f32, b: f32) -> f32 {
    wrap_angle(a - b).abs()
}

fn angle_distance_mod_pi(a: f32, b: f32) -> f32 {
    let difference = (a - b).rem_euclid(std::f32::consts::PI);
    difference.min(std::f32::consts::PI - difference)
}

fn subsample(points: &[EdgePoint], limit: usize) -> Vec<EdgePoint> {
    if limit == 0 || points.len() <= limit {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(limit);
    points.iter().step_by(stride).copied().collect()
}

fn is_in_search_region(region: Option<RotatedRect>, x: f64, y: f64) -> bool {
    region.is_none_or(|region| region.contains(x as f32, y as f32))
}

fn scale_bins(params: &SearchParams) -> Vec<f32> {
    let bins = params.scale_bins.max(1);
    let minimum = params.scale_min.max(1e-4);
    let maximum = params.scale_max.max(minimum);
    if bins == 1 {
        return vec![minimum];
    }
    (0..bins)
        .map(|index| {
            let t = index as f32 / (bins - 1) as f32;
            minimum * (maximum / minimum).powf(t)
        })
        .collect()
}

fn rotation_bins(params: &SearchParams) -> Vec<f32> {
    let bins = params.rotation_bins.max(1);
    let minimum = params.rotation_min_deg.to_radians();
    let maximum = params.rotation_max_deg.to_radians();
    if bins == 1 {
        return vec![minimum];
    }
    (0..bins)
        .map(|index| minimum + (maximum - minimum) * index as f32 / (bins - 1) as f32)
        .collect()
}

/// Oriented generalized Hough voting (README §9.3).
#[must_use]
pub fn hough_hypotheses(
    model: &Model,
    level: &TestLevel,
    level_index: usize,
    params: &SearchParams,
) -> Vec<Hypothesis> {
    hough_hypotheses_with_progress(model, level, level_index, params, &IgnoreProgress)
}

fn hough_hypotheses_with_progress(
    model: &Model,
    level: &TestLevel,
    level_index: usize,
    params: &SearchParams,
    progress: &dyn ProgressSink,
) -> Vec<Hypothesis> {
    let model_points = subsample(
        &model.features_at_level(level_index).edges.points,
        params.max_model_hough_points,
    );
    if model_points.len() < 4 || level.edges.points.len() < 4 {
        return Vec::new();
    }
    let test_points = subsample(&level.edges.points, params.max_test_hough_points);
    let rotations = rotation_bins(params);
    let scales = scale_bins(params);
    if rotations.is_empty() || scales.is_empty() {
        return Vec::new();
    }

    let width = level.gray.width;
    let height = level.gray.height;
    let max_radius = model_points
        .iter()
        .map(|point| (point.x * point.x + point.y * point.y).sqrt())
        .fold(0.0f32, f32::max);
    let gate_margin = (params.orientation_gate * 2.0).ceil() as i32;

    // A scale's votes reach at most `max_radius * scale` from the test point,
    // so each scale gets only the margin it needs.  That shrinks both the
    // reset (allocation/zero) and the peak scan, which dominate Hough cost.
    let layout: Vec<(i32, i32, i32)> = scales
        .iter()
        .map(|&scale| {
            let margin = (max_radius * scale).ceil() as i32 + gate_margin;
            (
                margin,
                width as i32 + 2 * margin,
                height as i32 + 2 * margin,
            )
        })
        .collect();
    if layout.iter().any(|&(_, w, h)| w <= 2 || h <= 2) {
        return Vec::new();
    }

    let orientation_buckets = HOUGH_ORIENTATION_BUCKETS;
    let bucket_width = std::f32::consts::TAU / orientation_buckets as f32;
    let gate = params.orientation_gate.clamp(0.01, std::f32::consts::PI);
    // A model point within `gate` of the target angle can lie at most
    // `floor(gate / bucket_width) + 1` buckets from the target's bucket; the
    // extra bucket also covers a point landing exactly on a bucket boundary.
    let reach = ((gate / bucket_width).floor() as i32 + 1).max(1);
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); orientation_buckets];
    let directed = params.polarity_enabled;
    let model_angles: Vec<f32> = model_points
        .iter()
        .map(|point| {
            let sign = if directed { f32::from(point.sign) } else { 1.0 };
            (point.ny * sign).atan2(point.nx * sign)
        })
        .collect();
    for (index, angle) in model_angles.iter().enumerate() {
        let bucket = ((angle.rem_euclid(std::f32::consts::TAU) / bucket_width).floor() as usize)
            .min(orientation_buckets - 1);
        buckets[bucket].push(index);
    }
    let test_angles: Vec<f32> = test_points
        .iter()
        .map(|point| {
            let sign = if directed { f32::from(point.sign) } else { 1.0 };
            (point.ny * sign).atan2(point.nx * sign)
        })
        .collect();
    let completed = AtomicUsize::new(0);
    let total_rotations = rotations.len().max(1);
    let max_accumulator_size = layout
        .iter()
        .map(|&(_, width, height)| width as usize * height as usize)
        .max()
        .unwrap_or(0);

    let per_rotation: Vec<Vec<Hypothesis>> = rotations
        .par_iter()
        .map_init(
            || HoughScratch::new(max_accumulator_size),
            |scratch, &theta| {
                if progress.cancelled() {
                    return Vec::new();
                }
                let (sin, cos) = theta.sin_cos();
                let HoughScratch {
                    accumulator,
                    entries,
                } = scratch;
                entries.clear();

                // Pass 1: resolve the orientation gate once per rotation and
                // remember the votes that survive it.  The vote order is the
                // same as the original nested loops, so each accumulator cell
                // sums its weights in the same order.
                for (test_index, test_point) in test_points.iter().enumerate() {
                    if test_index % 256 == 0 && progress.cancelled() {
                        return Vec::new();
                    }
                    let test_angle = test_angles[test_index];
                    let test_weight = test_point.weight.max(1e-3);
                    let target = (test_angle - theta).rem_euclid(std::f32::consts::TAU);
                    let center = (target / bucket_width).floor() as i32;
                    for bucket_offset in -reach..=reach {
                        let bucket = (center + bucket_offset).rem_euclid(orientation_buckets as i32)
                            as usize;
                        for &model_index in &buckets[bucket] {
                            let model_point = &model_points[model_index];
                            let difference =
                                wrap_angle(test_angle - model_angles[model_index] - theta);
                            if difference.abs() > gate {
                                continue;
                            }
                            let kernel =
                                0.5 + 0.5 * (std::f32::consts::PI * difference / gate).cos();
                            let weight = model_point.weight.max(0.0) * test_weight * kernel;
                            if weight <= 0.0 {
                                continue;
                            }
                            entries.push(VoteEntry {
                                test_index: test_index as u32,
                                rotated_x: cos * model_point.x - sin * model_point.y,
                                rotated_y: sin * model_point.x + cos * model_point.y,
                                weight,
                            });
                        }
                    }
                }

                // Pass 2: accumulate and scan one scale at a time with a
                // single reusable grid.  Every scale's working set fits in
                // cache instead of twelve grids competing for L2/L3.
                let mut found = Vec::new();
                for (scale_index, &scale) in scales.iter().enumerate() {
                    let (margin, width, height) = layout[scale_index];
                    accumulator.configure(margin, width, height);
                    for entry in entries.iter() {
                        let test_point = &test_points[entry.test_index as usize];
                        let vote_x = test_point.x - scale * entry.rotated_x;
                        let vote_y = test_point.y - scale * entry.rotated_y;
                        let x = (vote_x + margin as f32).round() as i32;
                        let y = (vote_y + margin as f32).round() as i32;
                        if x < 0 || y < 0 || x >= width || y >= height {
                            continue;
                        }
                        accumulator.vote(x, y, entry.weight);
                    }
                    let maximum = accumulator.maximum;
                    if maximum > 0.0 {
                        let threshold = maximum * params.peak_fraction.clamp(0.0, 1.0);
                        let mut peaks = accumulator.peaks(threshold);
                        peaks.sort_by(|a, b| {
                            b.0.total_cmp(&a.0)
                                .then_with(|| a.2.cmp(&b.2))
                                .then_with(|| a.1.cmp(&b.1))
                        });
                        for &(value, x, y) in peaks.iter().take(params.peaks_per_bin.max(1)) {
                            let translation_x = (x - margin) as f64;
                            let translation_y = (y - margin) as f64;
                            let factor = (1u32 << level_index) as f64;
                            let pose = Similarity::from_pose(
                                translation_x * factor,
                                translation_y * factor,
                                f64::from(theta),
                                f64::from(scale),
                            );
                            if is_in_search_region(params.search_region, pose.tx, pose.ty) {
                                found.push(Hypothesis {
                                    pose,
                                    seed_score: value,
                                    source: SeedSource::Hough,
                                    level: level_index,
                                });
                            }
                        }
                    }
                    accumulator.reset();
                }
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress.progress(
                    DetectStage::Search,
                    0.30 + 0.12 * done as f32 / total_rotations as f32,
                );
                found
            },
        )
        .collect();
    per_rotation.into_iter().flatten().collect()
}

/// One qualifying vote in a rotation's orientation-gated list.
#[derive(Clone, Copy)]
struct VoteEntry {
    test_index: u32,
    rotated_x: f32,
    rotated_y: f32,
    weight: f32,
}

/// Per-worker Hough scratch: the gated vote list and one reusable grid.
struct HoughScratch {
    accumulator: HoughAccumulator,
    entries: Vec<VoteEntry>,
}

impl HoughScratch {
    fn new(accumulator_size: usize) -> Self {
        Self {
            accumulator: HoughAccumulator::new(accumulator_size),
            entries: Vec::with_capacity(4096),
        }
    }
}

/// A Hough accumulator grid, reused across scales by a rotation worker.
///
/// The grid is reconfigured per scale: the margin covers that scale's maximum
/// vote reach, so scales use smaller grids than the worst-case (`scale_max`)
/// margin would require.  The voted bounding box drives the peak scan and the
/// reset, which touch only cells that can hold a vote.
struct HoughAccumulator {
    margin: i32,
    width: i32,
    height: i32,
    values: Vec<f32>,
    maximum: f32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

impl HoughAccumulator {
    fn new(capacity: usize) -> Self {
        Self {
            margin: 0,
            width: 0,
            height: 0,
            values: vec![0.0; capacity],
            maximum: 0.0,
            min_x: i32::MAX,
            max_x: i32::MIN,
            min_y: i32::MAX,
            max_y: i32::MIN,
        }
    }

    /// Point the accumulator at another scale's grid.
    ///
    /// The buffer is shared across scales and is already zero because the
    /// previous scale's [`Self::reset`] cleared every cell it voted into.
    fn configure(&mut self, margin: i32, width: i32, height: i32) {
        debug_assert!(width as usize * height as usize <= self.values.len());
        self.margin = margin;
        self.width = width;
        self.height = height;
    }

    /// Add `weight` to a cell and extend the voted bounding box.
    #[inline]
    fn vote(&mut self, x: i32, y: i32, weight: f32) {
        let index = y as usize * self.width as usize + x as usize;
        let value = self.values[index] + weight;
        self.values[index] = value;
        if value > self.maximum {
            self.maximum = value;
        }
        self.min_x = self.min_x.min(x);
        self.max_x = self.max_x.max(x);
        self.min_y = self.min_y.min(y);
        self.max_y = self.max_y.max(y);
    }

    /// Restore the accumulator to all zeros for the next scale.
    ///
    /// Every non-zero cell is inside the voted bounding box, so clearing the
    /// box is enough; margins that no vote reached are already zero.
    fn reset(&mut self) {
        if self.min_x <= self.max_x && self.min_y <= self.max_y {
            let width = self.width as usize;
            for y in self.min_y..=self.max_y {
                let start = y as usize * width + self.min_x as usize;
                let end = y as usize * width + self.max_x as usize + 1;
                self.values[start..end].fill(0.0);
            }
        }
        self.maximum = 0.0;
        self.min_x = i32::MAX;
        self.max_x = i32::MIN;
        self.min_y = i32::MAX;
        self.max_y = i32::MIN;
    }

    /// Local maxima at or above `threshold`, in raster order.
    ///
    /// Only the voted bounding box expanded by one cell can contain a
    /// qualifying peak, because every cell outside it is zero.  A zero
    /// threshold makes every cell a candidate, so the full interior is
    /// scanned to preserve the previous behaviour.
    fn peaks(&self, threshold: f32) -> Vec<(f32, i32, i32)> {
        let mut peaks = Vec::new();
        if self.maximum <= 0.0 {
            return peaks;
        }
        let (x0, x1, y0, y1) = if threshold <= 0.0 {
            (1, self.width - 2, 1, self.height - 2)
        } else {
            (
                (self.min_x - 1).max(1),
                (self.max_x + 1).min(self.width - 2),
                (self.min_y - 1).max(1),
                (self.max_y + 1).min(self.height - 2),
            )
        };
        if x0 > x1 || y0 > y1 {
            return peaks;
        }
        let width = self.width as usize;
        for y in y0..=y1 {
            let row = y as usize * width;
            let above = row - width;
            let below = row + width;
            for x in x0..=x1 {
                let column = x as usize;
                let index = row + column;
                let value = self.values[index];
                if value < threshold {
                    continue;
                }
                let local_max = self.values[above + column - 1] <= value
                    && self.values[above + column] <= value
                    && self.values[above + column + 1] <= value
                    && self.values[row + column - 1] <= value
                    && self.values[row + column + 1] <= value
                    && self.values[below + column - 1] <= value
                    && self.values[below + column] <= value
                    && self.values[below + column + 1] <= value;
                if local_max {
                    peaks.push((value, x, y));
                }
            }
        }
        peaks
    }
}

/// Corner-descriptor seeding (README §9.3).
#[must_use]
pub fn descriptor_hypotheses(
    model: &Model,
    levels: &[TestLevel],
    coarse_level: usize,
    params: &SearchParams,
) -> Vec<Hypothesis> {
    descriptor_hypotheses_with_progress(model, levels, coarse_level, params, &IgnoreProgress)
}

fn descriptor_hypotheses_with_progress(
    model: &Model,
    levels: &[TestLevel],
    coarse_level: usize,
    params: &SearchParams,
    progress: &dyn ProgressSink,
) -> Vec<Hypothesis> {
    if model.features.corners.is_empty() || params.model_corner_cap == 0 {
        return Vec::new();
    }
    let model_indices = top_corner_indices(&model.features.corners, params.model_corner_cap);
    let model_corners: Vec<Corner> = model_indices
        .iter()
        .map(|&index| model.features.corners[index])
        .collect();
    let model_descriptors: Vec<&CornerDescriptor> = model_indices
        .iter()
        .map(|&index| &model.descriptors[index])
        .collect();
    let mut hypotheses = Vec::new();

    let max_level = coarse_level.min(levels.len().saturating_sub(1));
    for (level, test_level) in levels.iter().enumerate().take(max_level + 1) {
        if progress.cancelled() {
            return Vec::new();
        }
        let progress_fraction = 0.42 + 0.10 * (level + 1) as f32 / (max_level + 1) as f32;
        if test_level.corners.is_empty() {
            progress.progress(DetectStage::Search, progress_fraction);
            continue;
        }
        let test_indices = top_corner_indices(&test_level.corners, params.test_corner_cap);
        let test_corners: Vec<Corner> = test_indices
            .iter()
            .map(|&index| test_level.corners[index])
            .collect();
        let test_descriptors = describe_corners(&test_corners, &test_level.edges);
        if test_descriptors.is_empty() {
            progress.progress(DetectStage::Search, progress_fraction);
            continue;
        }
        let rows = model_descriptors.len();
        let columns = test_descriptors.len();
        let distances: Vec<Vec<(f32, i32, i32)>> = (0..rows)
            .into_par_iter()
            .map(|row| {
                if progress.cancelled() {
                    return Vec::new();
                }
                (0..columns)
                    .map(|column| {
                        descriptor_distance(
                            &model_descriptors[row].values,
                            &test_descriptors[column].values,
                        )
                    })
                    .collect()
            })
            .collect();
        if progress.cancelled() {
            return Vec::new();
        }

        let mut best_test = vec![(f32::INFINITY, usize::MAX); columns];
        for (row, row_distances) in distances.iter().enumerate() {
            for (column, &(distance, _, _)) in row_distances.iter().enumerate() {
                if distance < best_test[column].0 {
                    best_test[column] = (distance, row);
                }
            }
        }

        for (row, row_distances) in distances.iter().enumerate() {
            let mut best = (f32::INFINITY, 0usize, (0, 0));
            let mut second = f32::INFINITY;
            for (column, &(distance, model_scale, test_scale)) in row_distances.iter().enumerate() {
                if distance < best.0 {
                    second = best.0;
                    best = (distance, column, (model_scale, test_scale));
                } else if distance < second {
                    second = distance;
                }
            }
            if !best.0.is_finite()
                || (second.is_finite() && best.0 >= params.descriptor_ratio * second)
            {
                continue;
            }
            let (_, column, (model_scale, test_scale)) = best;
            if best_test[column].1 != row {
                continue;
            }
            let (radius_test, radius_model) = (
                DESCRIPTOR_RADII[test_scale as usize],
                DESCRIPTOR_RADII[model_scale as usize],
            );
            let factor = (1u32 << level) as f32;
            let scale = factor * radius_test / radius_model.max(1e-6);
            if !(params.scale_min..=params.scale_max).contains(&scale) {
                continue;
            }
            let model_corner = &model_corners[row];
            let test_corner = &test_corners[column];
            let base_angle = wrap_angle(test_corner.orientation - model_corner.orientation);
            let test_full = Vector2::new(
                f64::from(test_corner.x * factor),
                f64::from(test_corner.y * factor),
            );
            let model_point = Vector2::new(f64::from(model_corner.x), f64::from(model_corner.y));
            let seed_score = 1.0 / (1.0 + best.0);
            for angle in [base_angle, wrap_angle(base_angle + std::f32::consts::PI)] {
                let rotation = Similarity::rotation(f64::from(angle));
                let translation = test_full - rotation * model_point * f64::from(scale);
                let pose = Similarity::from_pose(
                    translation.x,
                    translation.y,
                    f64::from(angle),
                    f64::from(scale),
                );
                if !(params.rotation_min_deg.to_radians()..=params.rotation_max_deg.to_radians())
                    .contains(&angle)
                {
                    continue;
                }
                if is_in_search_region(params.search_region, pose.tx, pose.ty) {
                    hypotheses.push(Hypothesis {
                        pose,
                        seed_score,
                        source: SeedSource::CornerDescriptor,
                        level,
                    });
                }
            }
        }
        progress.progress(DetectStage::Search, progress_fraction);
    }
    hypotheses
}

fn top_corner_indices(corners: &[Corner], limit: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..corners.len()).collect();
    indices.sort_by(|&a, &b| {
        corners[b]
            .weight
            .total_cmp(&corners[a].weight)
            .then_with(|| a.cmp(&b))
    });
    indices.truncate(limit);
    indices.sort_unstable();
    indices
}

fn descriptor_distance(model: &[f32], test: &[f32]) -> (f32, i32, i32) {
    let scales = DESCRIPTOR_RADII.len();
    let mut best = (f32::INFINITY, 0i32, 0i32);
    for model_scale in 0..scales {
        let model_start = model_scale * DESCRIPTOR_SCALE_DIM;
        if model_start + DESCRIPTOR_SCALE_DIM > model.len() {
            continue;
        }
        for delta in -1i32..=1 {
            let test_scale = model_scale as i32 + delta;
            if test_scale < 0 || test_scale >= scales as i32 {
                continue;
            }
            let test_start = test_scale as usize * DESCRIPTOR_SCALE_DIM;
            if test_start + DESCRIPTOR_SCALE_DIM > test.len() {
                continue;
            }
            let distance: f32 = model[model_start..model_start + DESCRIPTOR_SCALE_DIM]
                .iter()
                .zip(&test[test_start..test_start + DESCRIPTOR_SCALE_DIM])
                .map(|(a, b)| (a - b).abs())
                .sum();
            if distance < best.0 {
                best = (distance, model_scale as i32, test_scale);
            }
        }
    }
    best
}

/// Segment-pair invariant seeding (README §9.3).
#[must_use]
pub fn segment_hypotheses(
    model: &Model,
    levels: &[TestLevel],
    coarse_level: usize,
    params: &SearchParams,
) -> Vec<Hypothesis> {
    segment_hypotheses_with_progress(model, levels, coarse_level, params, &IgnoreProgress)
}

fn segment_hypotheses_with_progress(
    model: &Model,
    levels: &[TestLevel],
    coarse_level: usize,
    params: &SearchParams,
    progress: &dyn ProgressSink,
) -> Vec<Hypothesis> {
    if model.segment_pairs.is_empty() || model.features.segments.is_empty() {
        return Vec::new();
    }
    let mut hypotheses = Vec::new();
    let max_level = coarse_level.min(levels.len().saturating_sub(1));
    for (level, test_level) in levels.iter().enumerate().take(max_level + 1) {
        if progress.cancelled() {
            return Vec::new();
        }
        let progress_fraction = 0.52 + 0.06 * (level + 1) as f32 / (max_level + 1) as f32;
        if test_level.segments.len() < 2 {
            progress.progress(DetectStage::Search, progress_fraction);
            continue;
        }
        let test_pairs =
            segment_pair_invariants(&test_level.segments, params.segment_pair_cap.max(1));
        if test_pairs.is_empty() {
            progress.progress(DetectStage::Search, progress_fraction);
            continue;
        }
        let factor = (1u32 << level) as f32;
        for (model_pair_index, model_pair) in model.segment_pairs.iter().enumerate() {
            if model_pair_index % 32 == 0 && progress.cancelled() {
                return Vec::new();
            }
            let model_first = model.features.segments[model_pair.first];
            let model_second = model.features.segments[model_pair.second];
            for test_pair in &test_pairs {
                let test_first = test_level.segments[test_pair.first];
                let test_second = test_level.segments[test_pair.second];
                if (model_pair.length_ratio - test_pair.length_ratio).abs() > 0.15
                    || angle_distance_mod_pi(model_pair.angle_between, test_pair.angle_between)
                        > 0.15
                    || (model_pair.midpoint_distance - test_pair.midpoint_distance).abs() > 0.12
                    || angle_distance_mod_pi(model_pair.midpoint_angle, test_pair.midpoint_angle)
                        > 0.15
                {
                    continue;
                }
                let error = (model_pair.length_ratio - test_pair.length_ratio).abs() / 0.15
                    + angle_distance_mod_pi(model_pair.angle_between, test_pair.angle_between)
                        / 0.15
                    + (model_pair.midpoint_distance - test_pair.midpoint_distance).abs() / 0.12
                    + angle_distance_mod_pi(model_pair.midpoint_angle, test_pair.midpoint_angle)
                        / 0.15;
                let seed_score = 1.0 / (1.0 + error);
                derive_segment_pose(
                    model_first,
                    model_second,
                    test_first,
                    test_second,
                    factor,
                    seed_score,
                    level,
                    params,
                    &mut hypotheses,
                );
            }
        }
        progress.progress(DetectStage::Search, progress_fraction);
    }
    hypotheses
}

#[allow(clippy::too_many_arguments)]
fn derive_segment_pose(
    model_first: Segment,
    model_second: Segment,
    test_first: Segment,
    test_second: Segment,
    factor: f32,
    seed_score: f32,
    level: usize,
    params: &SearchParams,
    out: &mut Vec<Hypothesis>,
) {
    if model_first.length <= 1e-5 || model_second.length <= 1e-5 {
        return;
    }
    let first_scale = test_first.length * factor / model_first.length;
    let second_scale = test_second.length * factor / model_second.length;
    if !first_scale.is_finite() || !second_scale.is_finite() {
        return;
    }
    let ratio = (first_scale / second_scale).max(second_scale / first_scale);
    if ratio > 1.25 {
        return;
    }
    let scale = 0.5 * (first_scale + second_scale);
    if !(params.scale_min..=params.scale_max).contains(&scale) {
        return;
    }
    let angle_first = wrap_angle(test_first.orientation - model_first.orientation);
    let angle_second = wrap_angle(test_second.orientation - model_second.orientation);
    let half_turn = std::f32::consts::PI;
    let consistent = angle_distance(angle_first, angle_second) < 0.2
        || angle_distance(wrap_angle(angle_first + half_turn), angle_second) < 0.2;
    if !consistent {
        return;
    }
    let angle = angle_first;
    let model_midpoint = Vector2::new(
        f64::from((model_first.x0 + model_first.x1) * 0.5),
        f64::from((model_first.y0 + model_first.y1) * 0.5),
    );
    let test_midpoint = Vector2::new(
        f64::from((test_first.x0 + test_first.x1) * 0.5 * factor),
        f64::from((test_first.y0 + test_first.y1) * 0.5 * factor),
    );
    for branch in [angle, wrap_angle(angle + half_turn)] {
        if !(params.rotation_min_deg.to_radians()..=params.rotation_max_deg.to_radians())
            .contains(&branch)
        {
            continue;
        }
        let rotation = Similarity::rotation(f64::from(branch));
        let translation = test_midpoint - rotation * model_midpoint * f64::from(scale);
        let pose = Similarity::from_pose(
            translation.x,
            translation.y,
            f64::from(branch),
            f64::from(scale),
        );
        if is_in_search_region(params.search_region, pose.tx, pose.ty) {
            out.push(Hypothesis {
                pose,
                seed_score,
                source: SeedSource::SegmentPair,
                level,
            });
        }
    }
}

/// Greedy pose-space clustering (README §9.4).
#[must_use]
pub fn cluster_hypotheses(
    mut hypotheses: Vec<Hypothesis>,
    model_diagonal: f32,
    params: &SearchParams,
) -> Vec<Hypothesis> {
    hypotheses.sort_by(|a, b| {
        b.seed_score
            .total_cmp(&a.seed_score)
            .then_with(|| source_rank(a.source).cmp(&source_rank(b.source)))
            .then_with(|| a.pose.tx.total_cmp(&b.pose.tx))
            .then_with(|| a.pose.ty.total_cmp(&b.pose.ty))
            .then_with(|| a.pose.theta.total_cmp(&b.pose.theta))
            .then_with(|| a.pose.scale.total_cmp(&b.pose.scale))
    });
    let translation_tolerance = (0.08 * model_diagonal).clamp(3.0, 15.0) as f64;
    let angle_tolerance = 8.0f64.to_radians();
    let mut accepted: Vec<Hypothesis> = Vec::new();
    for hypothesis in hypotheses {
        if accepted.len() >= params.hypothesis_cap {
            break;
        }
        let duplicate = accepted.iter().any(|kept| {
            let distance =
                (kept.pose.tx - hypothesis.pose.tx).hypot(kept.pose.ty - hypothesis.pose.ty);
            let angle = (kept.pose.theta - hypothesis.pose.theta).abs();
            let angle = angle.min(std::f64::consts::TAU - angle);
            let scale_ratio = (kept.pose.scale / hypothesis.pose.scale)
                .max(hypothesis.pose.scale / kept.pose.scale);
            distance < translation_tolerance && angle < angle_tolerance && scale_ratio < 1.15
        });
        if !duplicate {
            accepted.push(hypothesis);
        }
    }
    accepted
}

fn pose_in_search_range(pose: &Similarity, params: &SearchParams) -> bool {
    if !pose.is_finite()
        || pose.scale < f64::from(params.scale_min)
        || pose.scale > f64::from(params.scale_max)
    {
        return false;
    }
    let minimum = params.rotation_min_deg.to_radians();
    let maximum = params.rotation_max_deg.to_radians();
    if maximum - minimum >= std::f32::consts::TAU - 1e-4 {
        return true;
    }
    let span = (maximum - minimum).max(0.0);
    let relative = wrap_angle(pose.theta as f32 - minimum);
    relative >= -1e-4 && relative <= span + 1e-4
}

fn source_rank(source: SeedSource) -> u8 {
    match source {
        SeedSource::Hough => 0,
        SeedSource::CornerDescriptor => 1,
        SeedSource::SegmentPair => 2,
    }
}

fn rank_hypotheses(
    hypotheses: Vec<Hypothesis>,
    level: &TestLevel,
    model_points: &[EdgePoint],
    level_index: usize,
    model_diagonal: f32,
    params: &SearchParams,
    progress: &dyn ProgressSink,
) -> Vec<Hypothesis> {
    if progress.cancelled() {
        return Vec::new();
    }
    let factor = (1u32 << level_index) as f64;
    let tau_d = params.score.tau_d / (1u32 << level_index) as f32;
    let completed = AtomicUsize::new(0);
    let total = hypotheses.len().max(1);
    let mut scored: Vec<(f32, Hypothesis)> = hypotheses
        .par_iter()
        .map(|hypothesis| {
            if progress.cancelled() {
                return (f32::NEG_INFINITY, *hypothesis);
            }
            let level_pose = Similarity::from_pose(
                hypothesis.pose.tx / factor,
                hypothesis.pose.ty / factor,
                hypothesis.pose.theta,
                hypothesis.pose.scale,
            );
            let score = score_similarity(
                model_points,
                level.gray.width,
                level.gray.height,
                &level.gradient,
                &level.distance,
                &level_pose,
                tau_d,
                &params.score,
            )
            .score;
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            progress.progress(
                DetectStage::Verify,
                0.58 + 0.10 * done as f32 / total as f32,
            );
            (score, *hypothesis)
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| source_rank(a.1.source).cmp(&source_rank(b.1.source)))
            .then_with(|| a.1.pose.tx.total_cmp(&b.1.pose.tx))
            .then_with(|| a.1.pose.ty.total_cmp(&b.1.pose.ty))
            .then_with(|| a.1.pose.theta.total_cmp(&b.1.pose.theta))
    });
    let best = scored.first().map_or(0.0, |(score, _)| *score);
    let floor = (best * 0.4).max(0.12);
    scored.retain(|(score, _)| *score >= floor);

    // Greedy diversity selection: spend the verification budget on distinct
    // occurrences rather than on several seeds for the same object.  A
    // higher-ranked hypothesis suppresses a lower-ranked one under the same
    // pose gates as detection NMS.  ROI overlap is deliberately omitted:
    // coarse seeds are unrefined, and a nearby false seed would otherwise
    // suppress the true seed for a real instance.
    let limit = params.max_hypotheses_scored.max(1);
    let mut selected: Vec<(f32, Hypothesis)> = Vec::with_capacity(limit.min(scored.len()));
    for (score, hypothesis) in scored {
        if selected.len() >= limit {
            break;
        }
        let duplicate = selected
            .iter()
            .any(|(_, kept)| coarse_seed_duplicate(kept, &hypothesis, model_diagonal, &params.nms));
        if !duplicate {
            selected.push((score, hypothesis));
        }
    }
    selected
        .into_iter()
        .map(|(_, hypothesis)| hypothesis)
        .collect()
}

/// Whether two coarse hypotheses predict the same occurrence.
///
/// The test mirrors the detection NMS pose gates: translation within
/// `distance_fraction` of the model diagonal at the larger scale, a directed
/// angle within `angle_threshold`, and a scale ratio below `1 + scale_ratio`.
fn coarse_seed_duplicate(
    a: &Hypothesis,
    b: &Hypothesis,
    model_diagonal: f32,
    nms: &NmsParams,
) -> bool {
    if !a.pose.is_finite() || !b.pose.is_finite() {
        return false;
    }
    let distance = (a.pose.tx - b.pose.tx).hypot(a.pose.ty - b.pose.ty);
    let scale = a.pose.scale.max(b.pose.scale).max(1e-3);
    if distance >= f64::from(nms.distance_fraction * model_diagonal) * scale {
        return false;
    }
    let angle = (a.pose.theta - b.pose.theta).rem_euclid(std::f64::consts::TAU);
    let angle = angle.min(std::f64::consts::TAU - angle);
    if angle >= f64::from(nms.angle_threshold) {
        return false;
    }
    let ratio = (a.pose.scale / b.pose.scale).max(b.pose.scale / a.pose.scale);
    ratio < f64::from(1.0 + nms.scale_ratio)
}

#[allow(clippy::too_many_arguments)]
fn verify_hypothesis(
    model: &Model,
    levels: &[TestLevel],
    model_levels: &[Vec<EdgePoint>],
    coarse_level: usize,
    index: usize,
    hypothesis: &Hypothesis,
    params: &SearchParams,
    progress: &dyn ProgressSink,
    completed: &AtomicUsize,
    total: usize,
) -> Option<Detection> {
    if progress.cancelled() {
        return None;
    }
    let mut pose = hypothesis.pose;
    for level in (0..=coarse_level).rev() {
        let Some(test_level) = levels.get(level) else {
            continue;
        };
        let Some(points) = model_levels.get(level) else {
            continue;
        };
        if points.is_empty() {
            continue;
        }
        let factor = (1u32 << level) as f64;
        let level_pose =
            Similarity::from_pose(pose.tx / factor, pose.ty / factor, pose.theta, pose.scale);
        let tau_d = params.score.tau_d / (1u32 << level) as f32;
        let mut refine_params = params.refine;
        refine_params.broad_fallback = level == coarse_level;
        if level == 0 {
            refine_params.iterations = refine_params.iterations.max(8);
        }
        let refined = refine_similarity_level(
            points,
            test_level.gray.width,
            test_level.gray.height,
            &test_level.gradient,
            &test_level.distance,
            level_pose,
            tau_d,
            &params.score,
            &refine_params,
        );
        pose = Similarity::from_pose(
            refined.pose.tx * factor,
            refined.pose.ty * factor,
            refined.pose.theta,
            refined.pose.scale,
        );
    }

    if progress.cancelled() {
        return None;
    }
    if !pose_in_search_range(&pose, params) {
        return None;
    }
    let test_level = levels.first()?;
    let model_points = model.features_at_level(0).edges.points;
    let similarity_score = score_similarity(
        &model_points,
        test_level.gray.width,
        test_level.gray.height,
        &test_level.gradient,
        &test_level.distance,
        &pose,
        params.score.tau_d,
        &params.score,
    );

    let roi_model = model_roi_in_model_coords(model);
    let input = HomographyInput {
        model_points: &model_points,
        model_corners: &model.features.corners,
        model_segments: &model.features.segments,
        roi_model: &roi_model,
        width: test_level.gray.width,
        height: test_level.gray.height,
        gradient: &test_level.gradient,
        distance: &test_level.distance,
        test_corners: &test_level.corners,
        test_segments: &test_level.segments,
        tau_d: params.score.tau_d,
        score_params: &params.score,
    };
    progress.progress(
        DetectStage::Homography,
        0.68 + 0.29 * completed.load(Ordering::Relaxed) as f32 / total as f32,
    );
    let outcome = refine_homography(&input, pose, &similarity_score, &params.homography);
    if progress.cancelled() {
        return None;
    }

    let (pose, homography, homography_estimated, score) = match outcome.homography {
        Some(homography) => {
            let local = homography.pose_at_origin()?;
            let pose = Similarity::from_pose(local.x, local.y, local.angle, local.scale);
            (pose, homography, true, outcome.score)
        }
        None => (pose, pose.embed(), false, similarity_score),
    };
    let fallback_reason = outcome
        .fallback_reason
        .map(|reason| reason.as_str().to_owned());
    let homography = matrix_to_array(&homography.normalized_for_output());
    let diagnostics = DetectionDiagnostics {
        coverage: score.coverage,
        matched_fraction: score.matched_fraction,
        mean_distance_px: score.mean_distance,
        mean_orientation_agreement: score.mean_orientation_agreement,
        seed_source: hypothesis.source,
        correspondences: outcome.correspondences,
        inliers: outcome.inliers,
        ransac_iterations: outcome.ransac_iterations,
        corner_correspondences: outcome.corner_correspondences,
        corner_inliers: outcome.corner_inliers,
        homography_condition: outcome.condition,
        ambiguity: 1.0,
        fallback_reason,
    };
    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
    progress.progress(
        DetectStage::Homography,
        0.68 + 0.29 * done as f32 / total as f32,
    );
    Some(Detection {
        id: index,
        pose,
        homography,
        homography_estimated,
        score: score.score,
        confidence: confidence_from(
            score.score,
            score.matched_fraction,
            score.mean_orientation_agreement,
            1.0,
            homography_estimated,
        ),
        diagnostics,
    })
}

/// Score helper re-exported for benchmark diagnostics.
#[must_use]
pub fn score_at_level(
    model: &Model,
    level: &TestLevel,
    level_index: usize,
    pose: &Similarity,
    params: &SearchParams,
) -> ScoreDiagnostics {
    let points = model.features_at_level(level_index).edges.points;
    let tau_d = params.score.tau_d / (1u32 << level_index) as f32;
    score_similarity(
        &points,
        level.gray.width,
        level.gray.height,
        &level.gradient,
        &level.distance,
        pose,
        tau_d,
        &params.score,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::geometry::ZeroFrame;
    use crate::gray::{GrayF32, LoadedImage};
    use crate::model::{Model, SearchDefaults, TeachSettings};

    #[derive(Default)]
    struct RecordingProgress {
        events: Mutex<Vec<(DetectStage, f32)>>,
        cancel_after: Option<usize>,
        polls: AtomicUsize,
    }

    impl ProgressSink for RecordingProgress {
        fn progress(&self, stage: DetectStage, fraction: f32) {
            self.events.lock().unwrap().push((stage, fraction));
        }

        fn cancelled(&self) -> bool {
            let polls = self.polls.fetch_add(1, Ordering::Relaxed) + 1;
            self.cancel_after.is_some_and(|limit| polls > limit)
        }
    }

    fn point_in_polygon(x: f32, y: f32, polygon: &[[f32; 2]]) -> bool {
        let mut inside = false;
        let mut previous = polygon.len() - 1;
        for current in 0..polygon.len() {
            let (xi, yi) = (polygon[current][0], polygon[current][1]);
            let (xj, yj) = (polygon[previous][0], polygon[previous][1]);
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            previous = current;
        }
        inside
    }

    fn source_image() -> GrayF32 {
        let width = 128;
        let height = 128;
        let polygon = [
            [38.0, 32.0],
            [86.0, 32.0],
            [86.0, 52.0],
            [58.0, 52.0],
            [58.0, 94.0],
            [38.0, 94.0],
        ];
        let hole = (72.0f32, 44.0f32, 6.0f32);
        let mut data = vec![0.12f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut covered = 0;
                let samples = 3;
                for sy in 0..samples {
                    for sx in 0..samples {
                        let fx = x as f32 + (sx as f32 + 0.5) / samples as f32;
                        let fy = y as f32 + (sy as f32 + 0.5) / samples as f32;
                        if point_in_polygon(fx, fy, &polygon)
                            && (fx - hole.0).hypot(fy - hole.1) > hole.2
                        {
                            covered += 1;
                        }
                    }
                }
                let coverage = covered as f32 / (samples * samples) as f32;
                data[y * width + x] = 0.12 + coverage * 0.78;
            }
        }
        GrayF32::new(width, height, data).unwrap()
    }

    fn test_model() -> Model {
        let gray = source_image();
        let display = gray.display_rgba();
        let loaded = LoadedImage { gray, display };
        let settings = TeachSettings {
            name: "test".to_owned(),
            roi: RotatedRect::new(62.0, 63.0, 88.0, 88.0, 0.0),
            zero: ZeroFrame {
                origin: [62.0, 63.0],
                angle: 0.0,
            },
            params: FeatureParams::default(),
            mask: Vec::new(),
            search_defaults: SearchDefaults::default(),
        };
        Model::build(&loaded, &settings).expect("model")
    }

    fn render_scene(
        source: &GrayF32,
        poses: &[Similarity],
        width: usize,
        height: usize,
    ) -> GrayF32 {
        let zero = [62.0f32, 63.0f32];
        let mut data = vec![0.12f32; width * height];
        for pose in poses {
            let (sin, cos) = (pose.theta as f32).sin_cos();
            let inverse_scale = 1.0 / pose.scale as f32;
            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - pose.tx as f32;
                    let dy = y as f32 - pose.ty as f32;
                    let px = (cos * dx + sin * dy) * inverse_scale + zero[0];
                    let py = (-sin * dx + cos * dy) * inverse_scale + zero[1];
                    let value = source.sample_bilinear(px, py);
                    if value > 0.3 {
                        data[y * width + x] = value;
                    }
                }
            }
        }
        GrayF32::new(width, height, data).unwrap()
    }

    #[test]
    fn detects_a_known_similarity_pose() {
        let model = test_model();
        assert!(
            model.features.edge_count() > 100,
            "edges {}",
            model.features.edge_count()
        );
        let pose = Similarity::from_pose(140.0, 120.0, 0.6, 1.2);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams::from(model.search_defaults);
        let result = detect(&model, &scene, &params).expect("detect");
        assert!(
            !result.detections.is_empty(),
            "no detections; generated {}, scored {}",
            result.hypotheses_generated,
            result.hypotheses_scored
        );
        let best = &result.detections[0];
        assert!((best.pose.tx - pose.tx).abs() < 1.0, "tx {}", best.pose.tx);
        assert!((best.pose.ty - pose.ty).abs() < 1.0, "ty {}", best.pose.ty);
        assert!(
            (best.pose.theta - pose.theta).abs() < 0.02,
            "theta {}",
            best.pose.theta
        );
        assert!(
            (best.pose.scale - pose.scale).abs() < 0.03,
            "scale {}",
            best.pose.scale
        );
    }

    #[test]
    fn detects_two_instances() {
        let model = test_model();
        let poses = [
            Similarity::from_pose(80.0, 70.0, 0.2, 0.8),
            Similarity::from_pose(180.0, 140.0, -1.1, 1.1),
        ];
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &poses, 256, 224);
        let params = SearchParams::from(model.search_defaults);
        let result = detect(&model, &scene, &params).expect("detect");
        for pose in poses {
            assert!(
                result.detections.iter().any(|detection| {
                    (detection.pose.tx - pose.tx).abs() < 2.0
                        && (detection.pose.ty - pose.ty).abs() < 2.0
                }),
                "missing instance at {pose:?}"
            );
        }
    }

    #[test]
    fn detection_is_deterministic() {
        let model = test_model();
        let pose = Similarity::from_pose(140.0, 120.0, -0.4, 1.0);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams::from(model.search_defaults);
        let first = detect(&model, &scene, &params).expect("detect");
        let second = detect(&model, &scene, &params).expect("detect");
        assert_eq!(first.detections, second.detections);
    }

    #[test]
    fn descriptor_and_segment_sources_detect_without_hough() {
        let model = test_model();
        let pose = Similarity::from_pose(130.0, 110.0, 0.7, 1.1);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams {
            hypothesis_sources: HypothesisSources::Descriptors,
            ..SearchParams::from(model.search_defaults)
        };
        let result = detect(&model, &scene, &params).expect("detect");
        assert!(
            !result.detections.is_empty(),
            "descriptor/segment sources found nothing"
        );
        let best = &result.detections[0];
        assert!((best.pose.tx - pose.tx).abs() < 2.0);
        assert!((best.pose.ty - pose.ty).abs() < 2.0);
    }

    #[test]
    fn search_region_rejects_far_translations() {
        let model = test_model();
        let pose = Similarity::from_pose(140.0, 120.0, 0.3, 1.0);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams {
            search_region: Some(RotatedRect::new(40.0, 40.0, 50.0, 50.0, 0.0)),
            ..SearchParams::from(model.search_defaults)
        };
        let result = detect(&model, &scene, &params).expect("detect");
        assert!(
            result.detections.is_empty(),
            "detection outside the search region"
        );
        let params = SearchParams {
            search_region: Some(RotatedRect::new(140.0, 120.0, 120.0, 120.0, 0.0)),
            ..SearchParams::from(model.search_defaults)
        };
        let result = detect(&model, &scene, &params).expect("detect");
        assert!(!result.detections.is_empty());
    }

    #[test]
    fn reloaded_model_produces_identical_detections() {
        let model = test_model();
        let pose = Similarity::from_pose(140.0, 120.0, 0.6, 1.2);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams::from(model.search_defaults);
        let first = detect(&model, &scene, &params).expect("detect");
        let path =
            std::env::temp_dir().join(format!("pm-m5-roundtrip-{}.pmodel", std::process::id()));
        model.save(&path).expect("save model");
        let loaded = Model::load(&path).expect("load model");
        let _ = std::fs::remove_file(&path);
        let second = detect(&loaded, &scene, &params).expect("detect loaded");
        assert_eq!(first.detections, second.detections);
    }

    #[test]
    fn similarity_scene_reports_the_similarity_fallback() {
        let model = test_model();
        let pose = Similarity::from_pose(140.0, 120.0, 0.6, 1.2);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams::from(model.search_defaults);
        let result = detect(&model, &scene, &params).expect("detect");
        assert!(!result.detections.is_empty());
        for detection in &result.detections {
            assert!(
                !detection.homography_estimated,
                "false perspective claim at {:?}: {:?}",
                detection.pose, detection.diagnostics.fallback_reason
            );
            assert!(
                detection.diagnostics.fallback_reason.is_some(),
                "fallback reason missing"
            );
        }
    }

    #[test]
    fn blank_scene_has_no_detections() {
        let model = test_model();
        let scene = GrayF32::filled(256, 224, 0.12);
        let params = SearchParams::from(model.search_defaults);
        let result = detect(&model, &scene, &params).expect("detect");
        assert!(result.detections.is_empty());
    }

    #[test]
    fn progress_hook_reports_stages_and_preserves_results() {
        let model = test_model();
        let pose = Similarity::from_pose(140.0, 120.0, 0.6, 1.2);
        let scene = render_scene(&model.source.as_ref().unwrap().gray, &[pose], 256, 224);
        let params = SearchParams::from(model.search_defaults);
        let sink = RecordingProgress::default();

        let controlled =
            detect_with_progress(&model, &scene, &params, &sink).expect("controlled detect");
        let plain = detect(&model, &scene, &params).expect("plain detect");
        assert_eq!(
            controlled.detections, plain.detections,
            "the progress hook must not change results"
        );
        assert_eq!(controlled.hypotheses_generated, plain.hypotheses_generated);
        assert_eq!(controlled.hypotheses_scored, plain.hypotheses_scored);

        let events = sink.events.lock().unwrap();
        for stage in [
            DetectStage::Preprocess,
            DetectStage::Edges,
            DetectStage::Search,
            DetectStage::Verify,
            DetectStage::Homography,
        ] {
            assert!(
                events.iter().any(|(reported, _)| *reported == stage),
                "missing stage {stage:?}: {events:?}"
            );
        }
        assert!(
            events
                .iter()
                .all(|(_, fraction)| fraction.is_finite() && (0.0..=1.0).contains(fraction)),
            "fractions outside [0, 1]: {events:?}"
        );
    }

    #[test]
    fn cancellation_aborts_with_a_cancelled_error() {
        let model = test_model();
        let scene = GrayF32::filled(256, 224, 0.12);
        let params = SearchParams::from(model.search_defaults);
        let sink = RecordingProgress {
            cancel_after: Some(2),
            ..RecordingProgress::default()
        };
        let result = detect_with_progress(&model, &scene, &params, &sink);
        assert!(
            matches!(result, Err(Error::Cancelled)),
            "expected cancellation, got {result:?}"
        );
    }

    /// Reference peak scan over the whole accumulator interior.
    fn full_scan_peaks(
        values: &[f32],
        width: usize,
        height: usize,
        threshold: f32,
    ) -> Vec<(f32, i32, i32)> {
        let mut peaks = Vec::new();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let value = values[y * width + x];
                if value < threshold {
                    continue;
                }
                let mut local_max = true;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let neighbour = (y as i32 + dy) as usize * width + (x as i32 + dx) as usize;
                        if values[neighbour] > value {
                            local_max = false;
                        }
                    }
                }
                if local_max {
                    peaks.push((value, x as i32, y as i32));
                }
            }
        }
        peaks
    }

    #[test]
    fn hough_accumulator_bbox_scan_matches_full_scan() {
        let mut accumulator = HoughAccumulator::new(32 * 32);
        accumulator.configure(0, 32, 32);
        for &(x, y, weight) in &[
            (10, 10, 1.0f32),
            (11, 10, 0.5),
            (10, 11, 0.5),
            (11, 11, 0.25),
            (20, 20, 2.0),
            (21, 20, 0.5),
            (25, 5, 0.75),
        ] {
            accumulator.vote(x, y, weight);
        }
        assert_eq!(
            accumulator.peaks(0.5),
            full_scan_peaks(&accumulator.values, 32, 32, 0.5)
        );
    }

    #[test]
    fn hough_accumulator_reset_clears_votes() {
        let mut accumulator = HoughAccumulator::new(16 * 16);
        accumulator.configure(2, 16, 16);
        accumulator.vote(5, 6, 1.0);
        accumulator.vote(7, 9, 2.0);
        accumulator.reset();
        assert!(accumulator.values.iter().all(|&value| value == 0.0));
        assert_eq!(accumulator.maximum, 0.0);
    }

    #[test]
    fn orientation_gate_modes_cover_the_rotation_step() {
        let params = SearchParams::default();
        assert_eq!(
            params.orientation_gate,
            SearchParams::FAST_ORIENTATION_GATE,
            "the default must stay the speed-oriented gate"
        );
        let rotations = rotation_bins(&params);
        assert!(rotations.len() >= 2);
        let step = rotations[1] - rotations[0];
        // Every pose angle lies within half a step of a bin, so a gate below
        // half the step would leave orientations that no bin accepts.
        for gate in [
            SearchParams::FAST_ORIENTATION_GATE,
            SearchParams::HIGH_ACCURACY_ORIENTATION_GATE,
        ] {
            assert!(
                2.0 * gate + 1e-6 >= step,
                "gate {gate} does not cover step {step}"
            );
        }
    }
}
