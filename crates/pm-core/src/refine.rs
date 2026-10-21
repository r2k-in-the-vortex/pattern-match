//! Correspondence refinement and RANSAC homography estimation.
//!
//! M4 provides the shared RANSAC machinery: deterministic sampling from a
//! seeded PRNG, degeneracy rejection, adaptive iteration counts, inlier
//! counting by symmetric transfer error, and refitting on the inlier set.
//!
//! M5 adds the normal-directed correspondence search and weighted similarity
//! Procrustes fit of README §9.6, with a score-based pattern-search fallback
//! and a fine final polish.
//!
//! M6 adds correspondence assembly, RANSAC with mixed normal-directed and
//! point residuals, robust Levenberg–Marquardt homography refinement with
//! re-association, and the validation/fallback gate of README §9.7.  The
//! similarity pose remains the reported fallback whenever the perspective
//! evidence is not strong enough.
//!
//! Theorem links:
//!
//! * R1 — [`HOMOGRAPHY_SAMPLE_SIZE`] samples exactly `k = 4` distinct
//!   correspondences; the all-inlier probability is the combinatorial ratio
//!   proved in the Lean `PatternMatch.RANSAC` module;
//! * R2 — [`required_iterations`] uses the conservative per-sample probability
//!   `w^k`; the Lean theorem bounds `C(I,k)/C(n,k)` by it;
//! * R3 — the loop stops once its accumulated success probability reaches the
//!   configured confidence, matching `1 - (1 - q)^N`;
//! * R4 — a higher inlier ratio lowers the required count; the Lean module
//!   proves the monotonicity statements;
//! * R5 — [`required_iterations_from_q`] is the `⌈log(1-P)/log(1-q)⌉` formula
//!   with the cap applied.

use nalgebra::{DMatrix, DVector, Matrix3, Vector2};
use serde::{Deserialize, Serialize};

use crate::feature::{Corner, EdgePoint, Segment};
use crate::geometry::{Homography, NUMERIC_EPS, Similarity, homogenize, normalized_dlt};
use crate::score::{ScoreDiagnostics, ScoreParams, score_homography};

/// Minimal sample size for a homography (four correspondences).
pub const HOMOGRAPHY_SAMPLE_SIZE: usize = 4;

/// RANSAC configuration (README §9.7, Appendix A).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RansacParams {
    /// Desired probability that at least one all-inlier sample is drawn.
    pub confidence: f64,
    /// Symmetric transfer-error threshold in pixels.
    pub inlier_threshold: f64,
    /// Hard cap on iterations.
    pub max_iterations: usize,
    /// Minimum number of inliers required to report a homography.
    pub min_inliers: usize,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            confidence: 0.99,
            inlier_threshold: 2.5,
            max_iterations: 5000,
            min_inliers: 6,
            seed: 0,
        }
    }
}

/// The result of a RANSAC run.
#[derive(Clone, Debug)]
pub struct RansacResult {
    /// The best validated homography, if the minimum inlier count was met.
    pub homography: Option<Homography>,
    /// Indices of the correspondences classified as inliers.
    pub inliers: Vec<usize>,
    /// Number of iterations actually executed.
    pub iterations: usize,
    /// Fraction of correspondences classified as inliers.
    pub inlier_ratio: f64,
}

/// Number of iterations needed for success probability `confidence` when one
/// minimal sample is all-inlier with probability `q` (R5).
///
/// Returns the cap when `q` is not in `(0, 1)`.
#[must_use]
pub fn required_iterations_from_q(confidence: f64, q: f64, cap: usize) -> usize {
    if !(confidence > 0.0 && confidence < 1.0) || q <= 0.0 {
        return cap;
    }
    if q >= 1.0 {
        return 1;
    }
    let required = (1.0 - confidence).ln() / (1.0 - q).ln();
    if !required.is_finite() || required <= 0.0 {
        return 1;
    }
    (required.ceil() as usize).clamp(1, cap)
}

/// Number of iterations needed from an inlier ratio `w` and sample size `k`
/// (R2/R5): the per-sample all-inlier probability is bounded by `w^k`.
#[must_use]
pub fn required_iterations(
    confidence: f64,
    inlier_ratio: f64,
    sample_size: usize,
    cap: usize,
) -> usize {
    required_iterations_from_q(confidence, inlier_ratio.powi(sample_size as i32), cap)
}

/// Exact iteration count by repeated multiplication, used to verify the log
/// formula against the Lean golden vectors (R5).
#[must_use]
pub fn required_iterations_exact(confidence: f64, q: f64, cap: usize) -> usize {
    if !(confidence > 0.0 && confidence < 1.0) || q <= 0.0 {
        return cap;
    }
    if q >= 1.0 {
        return 1;
    }
    let mut miss = 1.0_f64;
    for iteration in 0..=cap {
        if 1.0 - miss >= confidence {
            return iteration;
        }
        miss *= 1.0 - q;
    }
    cap
}

/// SplitMix64: a small deterministic PRNG for reproducible sampling.
#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// Draw `count` distinct indices in `0..n`.
    fn sample_distinct(&mut self, count: usize, n: usize) -> Vec<usize> {
        let mut chosen = Vec::with_capacity(count);
        while chosen.len() < count {
            let candidate = self.below(n);
            if !chosen.contains(&candidate) {
                chosen.push(candidate);
            }
        }
        chosen
    }
}

/// Symmetric transfer error under `h`, in pixels (README §9.7).
#[must_use]
pub fn symmetric_transfer_error(
    h: &Homography,
    inverse: Option<&Homography>,
    src: Vector2<f64>,
    dst: Vector2<f64>,
) -> f64 {
    let forward = h
        .try_act_point(src)
        .map_or(f64::INFINITY, |mapped| (mapped - dst).norm());
    let backward = inverse
        .and_then(|inv| inv.try_act_point(dst))
        .map_or(f64::INFINITY, |mapped| (mapped - src).norm());
    forward.max(backward)
}

fn count_inliers(
    h: &Homography,
    src: &[Vector2<f64>],
    dst: &[Vector2<f64>],
    threshold: f64,
) -> Vec<usize> {
    let inverse = h.try_inverse();
    src.iter()
        .zip(dst.iter())
        .enumerate()
        .filter_map(|(index, (p, q))| {
            let error = symmetric_transfer_error(h, inverse.as_ref(), *p, *q);
            (error <= threshold).then_some(index)
        })
        .collect()
}

/// Estimate a homography from correspondences with RANSAC (README §9.7).
///
/// The sampling sequence is fully determined by `params.seed`, so identical
/// inputs and parameters produce identical results.
#[must_use]
pub fn ransac_homography(
    src: &[Vector2<f64>],
    dst: &[Vector2<f64>],
    params: &RansacParams,
) -> RansacResult {
    let n = src.len().min(dst.len());
    if n < HOMOGRAPHY_SAMPLE_SIZE {
        return RansacResult {
            homography: None,
            inliers: Vec::new(),
            iterations: 0,
            inlier_ratio: 0.0,
        };
    }

    let mut rng = SplitMix64::new(params.seed);
    let mut best_homography: Option<Homography> = None;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut iterations = 0;
    let mut needed = params.max_iterations;

    for iteration in 0..params.max_iterations {
        if iteration >= needed {
            break;
        }
        iterations = iteration + 1;
        let sample = rng.sample_distinct(HOMOGRAPHY_SAMPLE_SIZE, n);
        let sample_correspondences: Vec<_> = sample
            .iter()
            .map(|&index| (src[index], dst[index]))
            .collect();
        let Some(h) = normalized_dlt(&sample_correspondences) else {
            continue;
        };
        let inliers = count_inliers(&h, src, dst, params.inlier_threshold);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_homography = Some(h);
            let ratio = best_inliers.len() as f64 / n as f64;
            needed = required_iterations(
                params.confidence,
                ratio,
                HOMOGRAPHY_SAMPLE_SIZE,
                params.max_iterations,
            )
            .max(1);
        }
    }

    // Refit on the full inlier set once, keeping the better of the two models.
    if best_homography.is_some() && best_inliers.len() >= HOMOGRAPHY_SAMPLE_SIZE {
        let inlier_correspondences: Vec<_> = best_inliers
            .iter()
            .map(|&index| (src[index], dst[index]))
            .collect();
        if let Some(refit) = normalized_dlt(&inlier_correspondences) {
            let refit_inliers = count_inliers(&refit, src, dst, params.inlier_threshold);
            if refit_inliers.len() >= best_inliers.len() {
                best_inliers = refit_inliers;
                best_homography = Some(refit);
            }
        }
    }

    let inlier_ratio = best_inliers.len() as f64 / n as f64;
    let homography = if best_inliers.len() >= params.min_inliers.max(HOMOGRAPHY_SAMPLE_SIZE) {
        best_homography
    } else {
        None
    };
    RansacResult {
        homography,
        inliers: best_inliers,
        iterations,
        inlier_ratio,
    }
}

/// One correspondence feeding homography estimation (README §9.7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correspondence {
    /// Model-frame point.
    pub model: Vector2<f64>,
    /// Matched image-frame point.
    pub image: Vector2<f64>,
    /// Image-frame unit normal for edge correspondences; `None` for corner and
    /// segment-endpoint correspondences.
    pub normal: Option<Vector2<f64>>,
    /// Relative confidence in `[0, 1]`.
    pub weight: f64,
}

/// Homography refinement and validation parameters (README §9.7, Appendix A).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HomographyParams {
    /// Whether to attempt a homography at all.
    pub enabled: bool,
    /// RANSAC configuration.
    pub ransac: RansacParams,
    /// Minimum similarity score before homography estimation is attempted.
    /// Candidates below this cannot plausibly reach the default detection
    /// threshold after a perspective upgrade.
    pub min_similarity_score: f32,
    /// Minimum assembled correspondences before RANSAC is attempted.
    pub min_correspondences: usize,
    /// Minimum edge correspondences used by the robust LM fit.
    pub min_lm_correspondences: usize,
    /// Normal-directed edge search radius, full-resolution pixels.
    pub edge_search_radius: f32,
    /// Maximum model edge points used per correspondence assembly.
    pub max_edge_correspondences: usize,
    /// Minimum gradient response for an edge correspondence.
    pub edge_response_min: f32,
    /// Corner correspondence gating radius, full-resolution pixels.
    pub corner_gate: f32,
    /// Corner undirected-orientation agreement gate, radians.
    pub corner_orientation_gate: f32,
    /// Maximum model corners used per assembly.
    pub max_corner_correspondences: usize,
    /// Minimum corner inliers when the model and test both have corners.
    pub min_corner_inliers: usize,
    /// Segment-endpoint gating radius, full-resolution pixels.
    pub segment_gate: f32,
    /// Segment undirected-orientation agreement gate, radians.
    pub segment_orientation_gate: f32,
    /// Maximum model segment endpoints used per assembly.
    pub max_segment_correspondences: usize,
    /// Cauchy scale for the robust LM residuals, pixels.
    pub robust_sigma: f32,
    /// LM iterations per re-association round.
    pub lm_iterations: usize,
    /// Re-association rounds.
    pub recombinations: usize,
    /// Maximum accepted matrix condition number.
    pub max_condition: f64,
    /// Minimum edge-score improvement over the similarity fallback.
    pub min_score_improvement: f32,
    /// Maximum relative change of the perspective terms across re-association.
    pub perspective_stability: f64,
}

impl Default for HomographyParams {
    fn default() -> Self {
        Self {
            enabled: true,
            ransac: RansacParams::default(),
            min_similarity_score: 0.45,
            min_correspondences: 12,
            min_lm_correspondences: 8,
            edge_search_radius: 3.0,
            max_edge_correspondences: 180,
            edge_response_min: 0.01,
            corner_gate: 8.0,
            corner_orientation_gate: 30.0f32.to_radians(),
            max_corner_correspondences: 80,
            min_corner_inliers: 8,
            segment_gate: 8.0,
            segment_orientation_gate: 20.0f32.to_radians(),
            max_segment_correspondences: 40,
            robust_sigma: 1.5,
            lm_iterations: 20,
            recombinations: 3,
            max_condition: 1.0e6,
            min_score_improvement: 0.01,
            perspective_stability: 0.35,
        }
    }
}

/// Why a homography was not used (README §9.7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomographyFallbackReason {
    /// Homography refinement is disabled by parameters.
    Disabled,
    /// The similarity score was below the attempt floor.
    LowSimilarityScore,
    /// Not enough assembled correspondences.
    InsufficientCorrespondences,
    /// RANSAC found no model meeting the inlier minimum.
    RansacFailed,
    /// The refined matrix became singular or non-finite.
    Degenerate,
    /// The matrix condition number exceeded the threshold.
    IllConditioned,
    /// A mapped ROI corner has non-positive homogeneous depth.
    DepthBehindCamera,
    /// Too few normal-directed edge correspondences for the robust fit.
    InsufficientEdgeCorrespondences,
    /// The robust LM fit failed numerically.
    LmFailed,
    /// Too few correspondences support the refined homography.
    TooFewInliers,
    /// Too few corner correspondences support the refined homography.
    TooFewCornerInliers,
    /// The edge score did not improve enough over the similarity fallback.
    ScoreNotImproved,
    /// The perspective terms were unstable across re-association.
    UnstablePerspective,
}

impl HomographyFallbackReason {
    /// Stable diagnostic name, used in result JSON.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::LowSimilarityScore => "low_similarity_score",
            Self::InsufficientCorrespondences => "insufficient_correspondences",
            Self::RansacFailed => "ransac_failed",
            Self::Degenerate => "degenerate_homography",
            Self::IllConditioned => "ill_conditioned",
            Self::DepthBehindCamera => "depth_behind_camera",
            Self::InsufficientEdgeCorrespondences => "insufficient_edge_correspondences",
            Self::LmFailed => "lm_failed",
            Self::TooFewInliers => "too_few_inliers",
            Self::TooFewCornerInliers => "too_few_corner_inliers",
            Self::ScoreNotImproved => "score_not_improved",
            Self::UnstablePerspective => "unstable_perspective",
        }
    }
}

/// The result of the homography stage for one detection.
#[derive(Clone, Debug)]
pub struct HomographyOutcome {
    /// Accepted homography, or `None` when the similarity fallback is used.
    pub homography: Option<Homography>,
    /// Score of the accepted homography, or the similarity score on fallback.
    pub score: ScoreDiagnostics,
    /// Number of correspondences assembled under the final geometry.
    pub correspondences: usize,
    /// Number of final correspondences classified as inliers.
    pub inliers: usize,
    /// Corner correspondences assembled under the final geometry.
    pub corner_correspondences: usize,
    /// Corner correspondences consistent with the final homography.
    pub corner_inliers: usize,
    /// RANSAC iterations executed.
    pub ransac_iterations: usize,
    /// Numerical condition number, when a homography was accepted.
    pub condition: Option<f64>,
    /// Reason the similarity fallback was used.
    pub fallback_reason: Option<HomographyFallbackReason>,
}

/// Inputs for [`refine_homography`] at full resolution.
pub struct HomographyInput<'a> {
    /// Model edge points in model coordinates.
    pub model_points: &'a [EdgePoint],
    /// Model corners in model coordinates.
    pub model_corners: &'a [Corner],
    /// Model segments in model coordinates.
    pub model_segments: &'a [Segment],
    /// Taught ROI corners in model coordinates.
    pub roi_model: &'a [[f32; 2]; 4],
    /// Test-level width.
    pub width: usize,
    /// Test-level height.
    pub height: usize,
    /// Test-level gradient field.
    pub gradient: &'a crate::gradient::GradientField,
    /// Test-level distance field.
    pub distance: &'a crate::edge::DistanceField,
    /// Test-level corners.
    pub test_corners: &'a [Corner],
    /// Test-level segments.
    pub test_segments: &'a [Segment],
    /// Level-scaled scoring distance tolerance.
    pub tau_d: f32,
    /// Scoring parameters.
    pub score_params: &'a ScoreParams,
}

/// Estimate, refine, and validate a homography starting from a similarity pose
/// (README §9.7).
///
/// Returns the accepted homography or a similarity fallback with the reason and
/// the diagnostics of the attempt.  The caller keeps the similarity pose when
/// [`HomographyOutcome::homography`] is `None`.
#[must_use]
pub fn refine_homography(
    input: &HomographyInput<'_>,
    pose: Similarity,
    similarity_score: &ScoreDiagnostics,
    params: &HomographyParams,
) -> HomographyOutcome {
    let mut outcome = HomographyOutcome {
        homography: None,
        score: *similarity_score,
        correspondences: 0,
        inliers: 0,
        corner_correspondences: 0,
        corner_inliers: 0,
        ransac_iterations: 0,
        condition: None,
        fallback_reason: None,
    };
    if !params.enabled {
        outcome.fallback_reason = Some(HomographyFallbackReason::Disabled);
        return outcome;
    }
    if similarity_score.score < params.min_similarity_score {
        outcome.fallback_reason = Some(HomographyFallbackReason::LowSimilarityScore);
        return outcome;
    }

    let embedded = pose.embed();
    let assembled = assemble_correspondences(input, &embedded, params);
    outcome.correspondences = assembled.len();
    if assembled.len() < params.min_correspondences.max(HOMOGRAPHY_SAMPLE_SIZE) {
        outcome.fallback_reason = Some(HomographyFallbackReason::InsufficientCorrespondences);
        return outcome;
    }

    let ransac = ransac_homography_correspondences(&assembled, &params.ransac);
    outcome.ransac_iterations = ransac.iterations;
    outcome.inliers = ransac.inliers.len();
    let Some(mut homography) = ransac.homography else {
        outcome.fallback_reason = Some(HomographyFallbackReason::RansacFailed);
        return outcome;
    };

    let minimum_edges = params.min_lm_correspondences.max(HOMOGRAPHY_SAMPLE_SIZE);
    let mut edge_correspondences: Vec<Correspondence> = assembled
        .iter()
        .filter(|correspondence| correspondence.normal.is_some())
        .copied()
        .collect();
    if edge_correspondences.len() < minimum_edges {
        outcome.fallback_reason = Some(HomographyFallbackReason::InsufficientEdgeCorrespondences);
        return outcome;
    }

    // Alternate robust LM and re-association.  The perspective terms recorded
    // after each round detect fits that move with the association set instead
    // of converging to stable evidence.
    let mut previous_perspective: Option<(f64, f64)> = None;
    let mut last_perspective_change = 0.0;
    for _ in 0..params.recombinations.max(1) {
        let Some(refined) = lm_homography(&homography, &edge_correspondences, params) else {
            outcome.fallback_reason = Some(HomographyFallbackReason::LmFailed);
            return outcome;
        };
        homography = refined;
        let perspective = perspective_terms(&homography);
        if let Some(previous) = previous_perspective {
            last_perspective_change = relative_perspective_change(previous, perspective);
        }
        previous_perspective = Some(perspective);
        let reassembled = assemble_edge_correspondences(input, &homography, params);
        if reassembled.len() < minimum_edges {
            outcome.fallback_reason =
                Some(HomographyFallbackReason::InsufficientEdgeCorrespondences);
            return outcome;
        }
        edge_correspondences = reassembled;
    }
    let Some(refined) = lm_homography(&homography, &edge_correspondences, params) else {
        outcome.fallback_reason = Some(HomographyFallbackReason::LmFailed);
        return outcome;
    };
    homography = refined;
    let perspective = perspective_terms(&homography);
    if let Some(previous) = previous_perspective {
        last_perspective_change = relative_perspective_change(previous, perspective);
    }
    let unstable = last_perspective_change > params.perspective_stability;

    let mut final_correspondences = assemble_edge_correspondences(input, &homography, params);
    let final_corners = assemble_corner_correspondences(input, &homography, params);
    outcome.corner_correspondences = final_corners.len();
    outcome.corner_inliers =
        count_correspondence_inliers(&homography, &final_corners, params.ransac.inlier_threshold)
            .len();
    final_correspondences.extend(final_corners);
    final_correspondences.extend(assemble_segment_correspondences(input, &homography, params));
    outcome.correspondences = final_correspondences.len();
    outcome.inliers = count_correspondence_inliers(
        &homography,
        &final_correspondences,
        params.ransac.inlier_threshold,
    )
    .len();
    if outcome.inliers < params.ransac.min_inliers.max(HOMOGRAPHY_SAMPLE_SIZE) {
        outcome.fallback_reason = Some(HomographyFallbackReason::TooFewInliers);
        return outcome;
    }
    if !input.model_corners.is_empty()
        && !input.test_corners.is_empty()
        && outcome.corner_inliers < params.min_corner_inliers
    {
        outcome.fallback_reason = Some(HomographyFallbackReason::TooFewCornerInliers);
        return outcome;
    }

    let condition = match validate_homography(&homography, input.roi_model, params) {
        Ok(condition) => condition,
        Err(reason) => {
            outcome.fallback_reason = Some(reason);
            return outcome;
        }
    };
    if unstable {
        outcome.fallback_reason = Some(HomographyFallbackReason::UnstablePerspective);
        return outcome;
    }
    let homography_score = score_homography(
        input.model_points,
        input.width,
        input.height,
        input.gradient,
        input.distance,
        &homography,
        input.tau_d,
        input.score_params,
    );
    if homography_score.score < similarity_score.score + params.min_score_improvement {
        outcome.fallback_reason = Some(HomographyFallbackReason::ScoreNotImproved);
        return outcome;
    }

    outcome.homography = Some(homography);
    outcome.score = homography_score;
    outcome.condition = Some(condition);
    outcome
}

/// Count correspondences consistent with `homography`.
///
/// Edge correspondences use the absolute normal-direction residual (tangential
/// slide along an edge is not an error); corner and segment-endpoint
/// correspondences use the symmetric transfer error.
#[must_use]
pub fn count_correspondence_inliers(
    homography: &Homography,
    correspondences: &[Correspondence],
    threshold: f64,
) -> Vec<usize> {
    let inverse = correspondences
        .iter()
        .any(|correspondence| correspondence.normal.is_none())
        .then(|| homography.try_inverse())
        .flatten();
    correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            let error = match correspondence.normal {
                Some(normal) => homography
                    .try_act_point(correspondence.model)
                    .map_or(f64::INFINITY, |mapped| {
                        ((mapped - correspondence.image).dot(&normal)).abs()
                    }),
                None => symmetric_transfer_error(
                    homography,
                    inverse.as_ref(),
                    correspondence.model,
                    correspondence.image,
                ),
            };
            (error <= threshold).then_some(index)
        })
        .collect()
}

/// RANSAC homography from mixed correspondences (README §9.7).
///
/// Minimal samples are four correspondences, rejected when a model or image
/// triple is collinear (D4).  Inlier counting uses normal-direction residuals
/// for edges and symmetric transfer error for point features.
#[must_use]
pub fn ransac_homography_correspondences(
    correspondences: &[Correspondence],
    params: &RansacParams,
) -> RansacResult {
    let n = correspondences.len();
    if n < HOMOGRAPHY_SAMPLE_SIZE {
        return RansacResult {
            homography: None,
            inliers: Vec::new(),
            iterations: 0,
            inlier_ratio: 0.0,
        };
    }

    let mut rng = SplitMix64::new(params.seed);
    let mut best_homography: Option<Homography> = None;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut iterations = 0;
    let mut needed = params.max_iterations;

    for iteration in 0..params.max_iterations {
        if iteration >= needed {
            break;
        }
        iterations = iteration + 1;
        let sample = rng.sample_distinct(HOMOGRAPHY_SAMPLE_SIZE, n);
        let sample_correspondences: Vec<_> = sample
            .iter()
            .map(|&index| (correspondences[index].model, correspondences[index].image))
            .collect();
        let Some(homography) = normalized_dlt(&sample_correspondences) else {
            continue;
        };
        let inliers =
            count_correspondence_inliers(&homography, correspondences, params.inlier_threshold);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_homography = Some(homography);
            let ratio = best_inliers.len() as f64 / n as f64;
            needed = required_iterations(
                params.confidence,
                ratio,
                HOMOGRAPHY_SAMPLE_SIZE,
                params.max_iterations,
            )
            .max(1);
        }
    }

    if best_homography.is_some() && best_inliers.len() >= HOMOGRAPHY_SAMPLE_SIZE {
        let inlier_correspondences: Vec<_> = best_inliers
            .iter()
            .map(|&index| (correspondences[index].model, correspondences[index].image))
            .collect();
        if let Some(refit) = normalized_dlt(&inlier_correspondences) {
            let refit_inliers =
                count_correspondence_inliers(&refit, correspondences, params.inlier_threshold);
            if refit_inliers.len() >= best_inliers.len() {
                best_inliers = refit_inliers;
                best_homography = Some(refit);
            }
        }
    }

    let inlier_ratio = best_inliers.len() as f64 / n as f64;
    let homography = if best_inliers.len() >= params.min_inliers.max(HOMOGRAPHY_SAMPLE_SIZE) {
        best_homography
    } else {
        None
    };
    RansacResult {
        homography,
        inliers: best_inliers,
        iterations,
        inlier_ratio,
    }
}

/// Validate a refined homography before reporting it (README §9.7).
///
/// Returns the condition number when the matrix is invertible, finite, inside
/// the condition threshold, and maps every ROI corner with positive depth.
pub fn validate_homography(
    homography: &Homography,
    roi_model: &[[f32; 2]; 4],
    params: &HomographyParams,
) -> std::result::Result<f64, HomographyFallbackReason> {
    if !homography.matrix().iter().all(|value| value.is_finite()) {
        return Err(HomographyFallbackReason::Degenerate);
    }
    let determinant = homography.determinant();
    if !determinant.is_finite() || determinant.abs() <= NUMERIC_EPS {
        return Err(HomographyFallbackReason::Degenerate);
    }
    let condition = homography.condition_number();
    if !condition.is_finite() || condition > params.max_condition {
        return Err(HomographyFallbackReason::IllConditioned);
    }
    let corners: Vec<Vector2<f64>> = roi_model
        .iter()
        .map(|corner| Vector2::new(f64::from(corner[0]), f64::from(corner[1])))
        .collect();
    if !homography.all_positive_depth(&corners)
        || corners
            .iter()
            .any(|corner| homography.try_act_point(*corner).is_none())
    {
        return Err(HomographyFallbackReason::DepthBehindCamera);
    }
    if homography.pose_at_origin().is_none() {
        return Err(HomographyFallbackReason::Degenerate);
    }
    Ok(condition)
}

fn perspective_terms(homography: &Homography) -> (f64, f64) {
    let matrix = homography.matrix();
    (matrix[(2, 0)], matrix[(2, 1)])
}

fn relative_perspective_change(previous: (f64, f64), current: (f64, f64)) -> f64 {
    let difference = (previous.0 - current.0)
        .abs()
        .max((previous.1 - current.1).abs());
    let scale = previous
        .0
        .abs()
        .max(previous.1.abs())
        .max(current.0.abs())
        .max(current.1.abs())
        .max(1e-6);
    difference / scale
}

fn assemble_correspondences(
    input: &HomographyInput<'_>,
    homography: &Homography,
    params: &HomographyParams,
) -> Vec<Correspondence> {
    let mut correspondences = assemble_edge_correspondences(input, homography, params);
    correspondences.extend(assemble_corner_correspondences(input, homography, params));
    correspondences.extend(assemble_segment_correspondences(input, homography, params));
    correspondences
}

fn assemble_edge_correspondences(
    input: &HomographyInput<'_>,
    homography: &Homography,
    params: &HomographyParams,
) -> Vec<Correspondence> {
    if input.model_points.is_empty() || input.width == 0 || input.height == 0 {
        return Vec::new();
    }
    let max_x = (input.width - 1) as f64;
    let max_y = (input.height - 1) as f64;
    let radius = f64::from(params.edge_search_radius.max(0.5));
    let step = 0.5f64;
    let steps = (radius / step).ceil() as i32;
    let limit = params.max_edge_correspondences.max(1);
    let stride = if input.model_points.len() <= limit {
        1
    } else {
        input.model_points.len().div_ceil(limit)
    };
    let response_min = f64::from(params.edge_response_min.max(0.0));
    let robust_sigma = f64::from(params.robust_sigma.max(0.1));
    let mut correspondences = Vec::new();

    for point in input.model_points.iter().step_by(stride) {
        let p = Vector2::new(f64::from(point.x), f64::from(point.y));
        let mapped = homography.act_homogeneous(homogenize(p));
        if !mapped.z.is_finite() || mapped.z <= NUMERIC_EPS {
            continue;
        }
        let q = Vector2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if q.x < 0.0 || q.y < 0.0 || q.x > max_x || q.y > max_y {
            continue;
        }
        let Some(normal) =
            homography.transform_normal(p, Vector2::new(f64::from(point.nx), f64::from(point.ny)))
        else {
            continue;
        };
        let response_at = |offset: f64| {
            let x = (q.x + offset * normal.x) as f32;
            let y = (q.y + offset * normal.y) as f32;
            let (gx, gy) = input.gradient.sample(x, y);
            let magnitude = f64::from(gx.hypot(gy));
            if magnitude <= 1e-9 {
                return 0.0f64;
            }
            let dot = (normal.x * f64::from(gx) + normal.y * f64::from(gy)) / magnitude;
            let agreement = (dot * f64::from(point.sign)).max(0.0);
            magnitude * agreement * agreement
        };

        let mut best_offset = 0.0f64;
        let mut best_response = 0.0f64;
        for index in -steps..=steps {
            let offset = f64::from(index) * step;
            let response = response_at(offset);
            if response > best_response {
                best_response = response;
                best_offset = offset;
            }
        }
        if best_response < response_min {
            continue;
        }
        let minus = response_at(best_offset - step);
        let center = response_at(best_offset);
        let plus = response_at(best_offset + step);
        let denominator = minus - 2.0 * center + plus;
        let delta = if denominator.abs() > 1e-12 {
            (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let offset = best_offset + delta * step;
        let robust = 1.0 / (1.0 + (offset / robust_sigma).powi(2));
        let weight = f64::from(point.weight.max(0.0)) * best_response * robust;
        if weight > 0.0 {
            correspondences.push(Correspondence {
                model: p,
                image: q + offset * normal,
                normal: Some(normal),
                weight,
            });
        }
    }
    correspondences
}

fn assemble_corner_correspondences(
    input: &HomographyInput<'_>,
    homography: &Homography,
    params: &HomographyParams,
) -> Vec<Correspondence> {
    if input.model_corners.is_empty() || input.test_corners.is_empty() {
        return Vec::new();
    }
    let max_response = input
        .test_corners
        .iter()
        .map(|corner| corner.response)
        .fold(0.0f32, f32::max);
    if max_response <= 0.0 {
        return Vec::new();
    }
    let max_x = (input.width.saturating_sub(1)) as f64;
    let max_y = (input.height.saturating_sub(1)) as f64;
    let gate = f64::from(params.corner_gate.max(0.0));
    let orientation_gate = f64::from(params.corner_orientation_gate.max(0.0));
    let mut correspondences = Vec::new();
    for corner in input
        .model_corners
        .iter()
        .take(params.max_corner_correspondences.max(1))
    {
        let p = Vector2::new(f64::from(corner.x), f64::from(corner.y));
        let mapped = homography.act_homogeneous(homogenize(p));
        if !mapped.z.is_finite() || mapped.z <= NUMERIC_EPS {
            continue;
        }
        let q = Vector2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if q.x < 0.0 || q.y < 0.0 || q.x > max_x || q.y > max_y {
            continue;
        }
        let Some(jacobian) = homography.jacobian_at(p) else {
            continue;
        };
        let Some(mapped_orientation) = mapped_orientation(&jacobian, f64::from(corner.orientation))
        else {
            continue;
        };
        let mut best: Option<(f64, &Corner)> = None;
        for test_corner in input.test_corners {
            let distance = (f64::from(test_corner.x) - q.x).hypot(f64::from(test_corner.y) - q.y);
            if distance > gate {
                continue;
            }
            let orientation_difference =
                orientation_distance_mod_pi(mapped_orientation, f64::from(test_corner.orientation));
            if orientation_difference > orientation_gate {
                continue;
            }
            if best.is_none_or(|(best_distance, _)| distance < best_distance) {
                best = Some((distance, test_corner));
            }
        }
        let Some((distance, test_corner)) = best else {
            continue;
        };
        let response_scale = f64::from((test_corner.response / max_response).clamp(0.0, 1.0));
        let proximity = (1.0 - distance / gate.max(1e-6)).max(0.0);
        let weight = (f64::from(corner.weight.max(0.0)) * response_scale).sqrt() * proximity;
        if weight > 0.0 {
            correspondences.push(Correspondence {
                model: p,
                image: Vector2::new(f64::from(test_corner.x), f64::from(test_corner.y)),
                normal: None,
                weight,
            });
        }
    }
    correspondences
}

fn assemble_segment_correspondences(
    input: &HomographyInput<'_>,
    homography: &Homography,
    params: &HomographyParams,
) -> Vec<Correspondence> {
    if input.model_segments.is_empty() || input.test_segments.is_empty() {
        return Vec::new();
    }
    let max_length = input
        .test_segments
        .iter()
        .map(|segment| segment.length)
        .fold(0.0f32, f32::max);
    if max_length <= 0.0 {
        return Vec::new();
    }
    let max_x = (input.width.saturating_sub(1)) as f64;
    let max_y = (input.height.saturating_sub(1)) as f64;
    let gate = f64::from(params.segment_gate.max(0.0));
    let orientation_gate = f64::from(params.segment_orientation_gate.max(0.0));
    let mut indices: Vec<usize> = (0..input.model_segments.len()).collect();
    indices.sort_by(|&a, &b| {
        input.model_segments[b]
            .weight
            .total_cmp(&input.model_segments[a].weight)
            .then_with(|| a.cmp(&b))
    });
    indices.truncate(params.max_segment_correspondences.max(1));
    let mut correspondences = Vec::new();
    for &index in &indices {
        let segment = input.model_segments[index];
        let direction = Vector2::new(
            f64::from(segment.x1 - segment.x0),
            f64::from(segment.y1 - segment.y0),
        );
        if direction.norm() <= NUMERIC_EPS {
            continue;
        }
        for endpoint in [
            Vector2::new(f64::from(segment.x0), f64::from(segment.y0)),
            Vector2::new(f64::from(segment.x1), f64::from(segment.y1)),
        ] {
            let mapped = homography.act_homogeneous(homogenize(endpoint));
            if !mapped.z.is_finite() || mapped.z <= NUMERIC_EPS {
                continue;
            }
            let q = Vector2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            if q.x < 0.0 || q.y < 0.0 || q.x > max_x || q.y > max_y {
                continue;
            }
            let Some(jacobian) = homography.jacobian_at(endpoint) else {
                continue;
            };
            let mapped_direction = jacobian * direction;
            if mapped_direction.norm() <= NUMERIC_EPS {
                continue;
            }
            let model_orientation = mapped_direction.y.atan2(mapped_direction.x);
            let mut best: Option<(f64, Vector2<f64>, f32)> = None;
            for test_segment in input.test_segments {
                if orientation_distance_mod_pi(
                    model_orientation,
                    f64::from(test_segment.orientation),
                ) > orientation_gate
                {
                    continue;
                }
                for point in [
                    Vector2::new(f64::from(test_segment.x0), f64::from(test_segment.y0)),
                    Vector2::new(f64::from(test_segment.x1), f64::from(test_segment.y1)),
                ] {
                    let distance = (point - q).norm();
                    if distance > gate {
                        continue;
                    }
                    if best.is_none_or(|(best_distance, _, _)| distance < best_distance) {
                        best = Some((distance, point, test_segment.length));
                    }
                }
            }
            let Some((distance, point, length)) = best else {
                continue;
            };
            let proximity = (1.0 - distance / gate.max(1e-6)).max(0.0);
            let weight = (f64::from(segment.weight.max(0.0))
                * f64::from((length / max_length).clamp(0.0, 1.0)))
            .sqrt()
                * proximity;
            if weight > 0.0 {
                correspondences.push(Correspondence {
                    model: endpoint,
                    image: point,
                    normal: None,
                    weight,
                });
            }
        }
    }
    correspondences
}

fn mapped_orientation(jacobian: &nalgebra::Matrix2<f64>, orientation: f64) -> Option<f64> {
    let direction = Vector2::new(orientation.cos(), orientation.sin());
    let mapped = jacobian * direction;
    if mapped.norm() <= NUMERIC_EPS {
        None
    } else {
        Some(mapped.y.atan2(mapped.x))
    }
}

fn orientation_distance_mod_pi(first: f64, second: f64) -> f64 {
    let difference = (first - second).rem_euclid(std::f64::consts::PI);
    difference.min(std::f64::consts::PI - difference)
}

fn matrix_from_params(parameters: &[f64; 8]) -> Matrix3<f64> {
    Matrix3::new(
        parameters[0],
        parameters[1],
        parameters[2],
        parameters[3],
        parameters[4],
        parameters[5],
        parameters[6],
        parameters[7],
        1.0,
    )
}

fn homography_loss(
    homography: &Matrix3<f64>,
    correspondences: &[Correspondence],
    sigma: f64,
) -> f64 {
    let mut loss = 0.0f64;
    for correspondence in correspondences {
        let Some(normal) = correspondence.normal else {
            continue;
        };
        let w = homography[(2, 0)] * correspondence.model.x
            + homography[(2, 1)] * correspondence.model.y
            + homography[(2, 2)];
        if !w.is_finite() || w <= NUMERIC_EPS {
            return f64::INFINITY;
        }
        let qx = (homography[(0, 0)] * correspondence.model.x
            + homography[(0, 1)] * correspondence.model.y
            + homography[(0, 2)])
            / w;
        let qy = (homography[(1, 0)] * correspondence.model.x
            + homography[(1, 1)] * correspondence.model.y
            + homography[(1, 2)])
            / w;
        let residual =
            (qx - correspondence.image.x) * normal.x + (qy - correspondence.image.y) * normal.y;
        let ratio = residual / sigma;
        loss += correspondence.weight.max(0.0) * sigma * sigma * (ratio * ratio).ln_1p();
    }
    loss
}

/// Robust Levenberg–Marquardt fit of the homography to normal-directed edge
/// correspondences (README §9.7).
///
/// The eight parameters are the affine-normalized entries with `h33 = 1`; a
/// Cauchy loss down-weights residual outliers in pixels.
fn lm_homography(
    start: &Homography,
    correspondences: &[Correspondence],
    params: &HomographyParams,
) -> Option<Homography> {
    let normalized = start.normalized_for_output();
    if !normalized.iter().all(|value| value.is_finite()) || (normalized[(2, 2)] - 1.0).abs() > 1e-9
    {
        return None;
    }
    let sigma = f64::from(params.robust_sigma.max(0.1));
    let mut current = normalized;
    let mut current_loss = homography_loss(&current, correspondences, sigma);
    if !current_loss.is_finite() {
        return None;
    }
    let mut lambda = 1e-3;

    for _ in 0..params.lm_iterations {
        let mut normal_matrix = DMatrix::<f64>::zeros(8, 8);
        let mut gradient = DVector::<f64>::zeros(8);
        let mut used = 0usize;
        for correspondence in correspondences {
            let Some(normal) = correspondence.normal else {
                continue;
            };
            let p = correspondence.model;
            let w = current[(2, 0)] * p.x + current[(2, 1)] * p.y + current[(2, 2)];
            if !w.is_finite() || w <= NUMERIC_EPS {
                continue;
            }
            let qx = (current[(0, 0)] * p.x + current[(0, 1)] * p.y + current[(0, 2)]) / w;
            let qy = (current[(1, 0)] * p.x + current[(1, 1)] * p.y + current[(1, 2)]) / w;
            let residual =
                (qx - correspondence.image.x) * normal.x + (qy - correspondence.image.y) * normal.y;
            let robust = 1.0 / (1.0 + (residual / sigma).powi(2));
            let weight = correspondence.weight.max(0.0) * robust;
            if weight <= 1e-12 {
                continue;
            }
            used += 1;
            let projected = normal.x * qx + normal.y * qy;
            let row = [
                normal.x * p.x / w,
                normal.x * p.y / w,
                normal.x / w,
                normal.y * p.x / w,
                normal.y * p.y / w,
                normal.y / w,
                -projected * p.x / w,
                -projected * p.y / w,
            ];
            for row_index in 0..8 {
                gradient[row_index] += weight * row[row_index] * residual;
                for column_index in 0..8 {
                    normal_matrix[(row_index, column_index)] +=
                        weight * row[row_index] * row[column_index];
                }
            }
        }
        if used < HOMOGRAPHY_SAMPLE_SIZE {
            break;
        }

        let mut system = normal_matrix.clone();
        for index in 0..8 {
            system[(index, index)] += lambda * normal_matrix[(index, index)].max(1e-12);
        }
        let Some(step) = system.lu().solve(&(-gradient)) else {
            lambda = (lambda * 10.0).min(1e12);
            continue;
        };
        if !step.iter().all(|value| value.is_finite()) {
            break;
        }
        let mut parameters = [0.0f64; 8];
        for (index, parameter) in parameters.iter_mut().enumerate() {
            *parameter = current[(index / 3, index % 3)] + step[index];
        }
        let candidate = matrix_from_params(&parameters);
        let candidate_loss = homography_loss(&candidate, correspondences, sigma);
        if candidate_loss.is_finite() && candidate_loss < current_loss {
            current = candidate;
            current_loss = candidate_loss;
            lambda = (lambda * 0.3).max(1e-12);
            if step.norm() < 1e-9 {
                break;
            }
        } else {
            lambda = (lambda * 10.0).min(1e12);
            if lambda >= 1e12 {
                break;
            }
        }
    }
    Some(Homography(current))
}

/// Parameters for similarity refinement (README §9.6).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimilarityRefineParams {
    /// Normal-directed correspondence search radius, in level pixels.
    pub search_radius: f32,
    /// ICP iterations per level.
    pub iterations: usize,
    /// Minimum reliable correspondences for a Procrustes fit.
    pub min_correspondences: usize,
    /// Robust weight scale for correspondence residuals.
    pub robust_sigma: f32,
    /// Minimum gradient response for a correspondence.
    pub response_min: f32,
    /// Use the broad 5⁴ multi-start grid when ICP fails.  The caller enables
    /// this only at the coarsest level.
    pub broad_fallback: bool,
}

impl Default for SimilarityRefineParams {
    fn default() -> Self {
        Self {
            search_radius: 3.5,
            iterations: 4,
            min_correspondences: 8,
            robust_sigma: 1.5,
            response_min: 0.01,
            broad_fallback: true,
        }
    }
}

/// Result of one level's similarity refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimilarityRefineResult {
    /// Refined level-local pose.
    pub pose: Similarity,
    /// Number of reliable correspondences used by the final fit.
    pub correspondences: usize,
}

/// Refine a similarity pose at one level (README §9.6).
///
/// Correspondences are searched along the transformed normal and fitted with a
/// weighted similarity Procrustes solve.  When too few correspondences survive,
/// the caller's pattern-search fallback is used.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn refine_similarity_level(
    model_points: &[crate::feature::EdgePoint],
    width: usize,
    height: usize,
    gradient: &crate::gradient::GradientField,
    distance: &crate::edge::DistanceField,
    pose: Similarity,
    tau_d: f32,
    score_params: &crate::score::ScoreParams,
    params: &SimilarityRefineParams,
) -> SimilarityRefineResult {
    let mut current = pose;
    let mut correspondences = 0usize;
    for _ in 0..params.iterations {
        let Some((next, count)) = icp_step(model_points, width, height, gradient, &current, params)
        else {
            break;
        };
        if !next.is_finite() || next.scale <= 0.02 {
            break;
        }
        current = next;
        correspondences = count;
    }

    if correspondences < params.min_correspondences {
        let polished = pattern_search_similarity(
            model_points,
            width,
            height,
            gradient,
            distance,
            &current,
            tau_d,
            score_params,
            &PatternSearchParams {
                iterations: 10,
                translation_step: 2.0,
                angle_step: 0.03,
                scale_step: 0.05,
                grid: params.broad_fallback,
                grid_radius: 1,
                max_sweeps: 4,
            },
        );
        current = polished;
    } else {
        current = pattern_search_similarity(
            model_points,
            width,
            height,
            gradient,
            distance,
            &current,
            tau_d,
            score_params,
            &PatternSearchParams {
                iterations: 3,
                translation_step: 0.25,
                angle_step: 0.002,
                scale_step: 0.002,
                grid: false,
                grid_radius: 1,
                max_sweeps: 3,
            },
        );
    }

    SimilarityRefineResult {
        pose: current,
        correspondences,
    }
}

fn icp_step(
    model_points: &[crate::feature::EdgePoint],
    width: usize,
    height: usize,
    gradient: &crate::gradient::GradientField,
    pose: &Similarity,
    params: &SimilarityRefineParams,
) -> Option<(Similarity, usize)> {
    if model_points.is_empty() || width == 0 || height == 0 {
        return None;
    }
    let rotation = Similarity::rotation(pose.theta);
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;
    let radius = f64::from(params.search_radius.max(0.5));
    let step = 0.5f32;
    let steps = (radius / f64::from(step)).ceil() as i32;

    let mut samples = Vec::new();
    for point in model_points {
        let p = Vector2::new(f64::from(point.x), f64::from(point.y));
        let q = pose.transform_point(p);
        if q.x < 0.0 || q.y < 0.0 || q.x > max_x || q.y > max_y {
            continue;
        }
        let normal = rotation * Vector2::new(f64::from(point.nx), f64::from(point.ny));
        let response_at = |offset: f32| {
            let x = (q.x + f64::from(offset) * normal.x) as f32;
            let y = (q.y + f64::from(offset) * normal.y) as f32;
            let (gx, gy) = gradient.sample(x, y);
            let magnitude = f64::from(gx.hypot(gy));
            if magnitude <= 1e-9 {
                return 0.0f64;
            }
            let dot = (normal.x * f64::from(gx) + normal.y * f64::from(gy)) / magnitude;
            let agreement = (dot * f64::from(point.sign)).max(0.0);
            magnitude * agreement * agreement
        };

        let mut best_offset = 0.0f32;
        let mut best_response = 0.0f64;
        for index in -steps..=steps {
            let offset = index as f32 * step;
            let response = response_at(offset);
            if response > best_response {
                best_response = response;
                best_offset = offset;
            }
        }
        if best_response < f64::from(params.response_min) {
            continue;
        }
        let minus = response_at(best_offset - step);
        let center = response_at(best_offset);
        let plus = response_at(best_offset + step);
        let denominator = minus - 2.0 * center + plus;
        let delta = if denominator.abs() > 1e-12 {
            let value = 0.5 * (minus - plus) / denominator;
            value.clamp(-0.5, 0.5) as f32
        } else {
            0.0
        };
        let offset = best_offset + delta * step;
        let correspondence = q + f64::from(offset) * normal;
        let robust =
            1.0 / (1.0 + (f64::from(offset) / f64::from(params.robust_sigma.max(0.1))).powi(2));
        let weight = f64::from(point.weight.max(0.0)) * best_response * robust;
        if weight > 0.0 {
            samples.push((p, correspondence, weight));
        }
    }

    if samples.len() < params.min_correspondences {
        return None;
    }
    let mut weight_sum = 0.0;
    let mut mean_p = Vector2::new(0.0, 0.0);
    let mut mean_q = Vector2::new(0.0, 0.0);
    for (p, q, weight) in &samples {
        weight_sum += weight;
        mean_p += *p * *weight;
        mean_q += *q * *weight;
    }
    if weight_sum <= 1e-12 {
        return None;
    }
    mean_p /= weight_sum;
    mean_q /= weight_sum;

    let mut dot: f64 = 0.0;
    let mut cross: f64 = 0.0;
    let mut norm: f64 = 0.0;
    for (p, q, weight) in &samples {
        let dp = p - mean_p;
        let dq = q - mean_q;
        dot += weight * (dp.x * dq.x + dp.y * dq.y);
        cross += weight * (dp.x * dq.y - dp.y * dq.x);
        norm += weight * dp.dot(&dp);
    }
    if norm <= 1e-12 {
        return None;
    }
    let scale = dot.hypot(cross) / norm;
    if !scale.is_finite() || scale <= 1e-4 {
        return None;
    }
    let theta = cross.atan2(dot);
    let rotated = Similarity::rotation(theta) * mean_p * scale;
    let translated = mean_q - rotated;
    Some((
        Similarity::from_pose(translated.x, translated.y, theta, scale),
        samples.len(),
    ))
}

/// Pattern-search parameters for the score-based fallback.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternSearchParams {
    /// Coordinate-descent sweeps.
    pub iterations: usize,
    /// Initial translation step in level pixels.
    pub translation_step: f32,
    /// Initial rotation step in radians.
    pub angle_step: f32,
    /// Initial relative scale step.
    pub scale_step: f32,
    /// Whether to seed coordinate descent with a parameter grid.
    pub grid: bool,
    /// Grid radius in steps (1 gives 3⁴, 2 gives 5⁴ candidates).
    pub grid_radius: i32,
    /// Maximum coordinate-descent sweeps per iteration.
    pub max_sweeps: usize,
}

impl Default for PatternSearchParams {
    fn default() -> Self {
        Self {
            iterations: 12,
            translation_step: 1.0,
            angle_step: 0.0175,
            scale_step: 0.02,
            grid: false,
            grid_radius: 2,
            max_sweeps: 8,
        }
    }
}

/// Maximize the edge score by coordinate descent around `pose` (README §9.6).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn pattern_search_similarity(
    model_points: &[crate::feature::EdgePoint],
    width: usize,
    height: usize,
    gradient: &crate::gradient::GradientField,
    distance: &crate::edge::DistanceField,
    pose: &Similarity,
    tau_d: f32,
    score_params: &crate::score::ScoreParams,
    params: &PatternSearchParams,
) -> Similarity {
    let mut best = *pose;
    let mut best_score = crate::score::score_similarity(
        model_points,
        width,
        height,
        gradient,
        distance,
        &best,
        tau_d,
        score_params,
    )
    .score;

    if params.grid {
        // A multi-start grid avoids the local maximum formed by a single
        // strong edge direction (the vertical-edge basin in a rectangle, for
        // example), which pure coordinate descent can enter and never leave.
        for dtheta in (-params.grid_radius..=params.grid_radius).map(|step| step as f32) {
            for dscale in (-params.grid_radius..=params.grid_radius).map(|step| step as f32) {
                for dy in (-params.grid_radius..=params.grid_radius).map(|step| step as f32) {
                    for dx in (-params.grid_radius..=params.grid_radius).map(|step| step as f32) {
                        if dtheta == 0.0 && dscale == 0.0 && dy == 0.0 && dx == 0.0 {
                            continue;
                        }
                        let candidate = Similarity::from_pose(
                            pose.tx + f64::from(dx * params.translation_step),
                            pose.ty + f64::from(dy * params.translation_step),
                            pose.theta + f64::from(dtheta * params.angle_step),
                            pose.scale * (1.0 + f64::from(dscale * params.scale_step)),
                        );
                        if candidate.scale <= 0.02 {
                            continue;
                        }
                        let score = crate::score::score_similarity(
                            model_points,
                            width,
                            height,
                            gradient,
                            distance,
                            &candidate,
                            tau_d,
                            score_params,
                        )
                        .score;
                        if score > best_score + 1e-6 {
                            best = candidate;
                            best_score = score;
                        }
                    }
                }
            }
        }
    }

    let mut steps = [
        params.translation_step,
        params.translation_step,
        params.angle_step,
        params.scale_step,
    ];
    for _ in 0..params.iterations {
        let mut improved = true;
        let mut sweeps = 0;
        while improved && sweeps < params.max_sweeps {
            sweeps += 1;
            improved = false;
            for (axis, step) in steps.iter().enumerate() {
                for direction in [-1.0f32, 1.0] {
                    let delta = direction * *step;
                    let candidate = perturb(&best, axis, delta);
                    if !candidate.is_finite() || candidate.scale <= 0.02 {
                        continue;
                    }
                    let score = crate::score::score_similarity(
                        model_points,
                        width,
                        height,
                        gradient,
                        distance,
                        &candidate,
                        tau_d,
                        score_params,
                    )
                    .score;
                    if score > best_score + 1e-6 {
                        best = candidate;
                        best_score = score;
                        improved = true;
                    }
                }
            }
        }
        for step in &mut steps {
            *step *= 0.6;
        }
    }
    best
}

fn perturb(pose: &Similarity, axis: usize, delta: f32) -> Similarity {
    let mut next = *pose;
    match axis {
        0 => next.tx += f64::from(delta),
        1 => next.ty += f64::from(delta),
        2 => next.theta += f64::from(delta),
        _ => next.scale *= 1.0 + f64::from(delta),
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::has_degenerate_triple;
    use nalgebra::Matrix3;

    fn test_homography() -> Homography {
        Homography::from_matrix(Matrix3::new(
            1.2, -0.3, 40.0, 0.4, 0.9, -25.0, 0.0002, -0.0001, 1.0,
        ))
    }

    fn inlier_correspondences(
        h: &Homography,
        count: usize,
        seed: u64,
    ) -> (Vec<Vector2<f64>>, Vec<Vector2<f64>>) {
        let mut rng = SplitMix64::new(seed);
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for _ in 0..count {
            let p = Vector2::new(rng.below(400) as f64 + 0.5, rng.below(300) as f64 + 0.5);
            src.push(p);
            dst.push(h.try_act_point(p).expect("finite homography"));
        }
        (src, dst)
    }

    #[test]
    fn required_iterations_match_the_log_formula_r5() {
        let cases = [
            (0.99, 0.5, 1000, 7),
            (0.999, 0.5, 1000, 10),
            (0.99, 0.9, 1000, 2),
            (0.99, 0.25, 1000, 17),
        ];
        for (confidence, q, cap, expected) in cases {
            assert_eq!(
                required_iterations_from_q(confidence, q, cap),
                expected,
                "log formula for confidence {confidence}, q {q}"
            );
            assert_eq!(
                required_iterations_exact(confidence, q, cap),
                expected,
                "exact loop for confidence {confidence}, q {q}"
            );
        }
        assert_eq!(required_iterations_from_q(0.99, 0.0, 1000), 1000);
        assert_eq!(required_iterations_from_q(0.99, 1.0, 1000), 1);
    }

    #[test]
    fn required_iterations_uses_conservative_power_r2() {
        // w = 0.625, k = 4: the per-sample bound is w^4 = 0.152587890625.
        let expected = required_iterations_from_q(0.99, 0.625_f64.powi(4), 5000);
        assert_eq!(required_iterations(0.99, 0.625, 4, 5000), expected);
    }

    #[test]
    fn ransac_recovers_exact_homography_r3() {
        let h = test_homography();
        let (src, dst) = inlier_correspondences(&h, 24, 7);
        let params = RansacParams {
            inlier_threshold: 1e-6,
            min_inliers: 10,
            seed: 42,
            ..RansacParams::default()
        };
        let result = ransac_homography(&src, &dst, &params);
        let recovered = result.homography.expect("exact inliers recover H");
        assert_eq!(result.inliers.len(), src.len());
        for i in 0..3 {
            for j in 0..3 {
                let actual = recovered.normalized_for_output()[(i, j)];
                let expected = h.normalized_for_output()[(i, j)];
                assert!(
                    (actual - expected).abs() <= 1e-8,
                    "R3 recovered H[{i},{j}]: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn ransac_ignores_outliers() {
        let h = test_homography();
        let (mut src, mut dst) = inlier_correspondences(&h, 30, 11);
        let mut rng = SplitMix64::new(99);
        for _ in 0..20 {
            let p = Vector2::new(rng.below(400) as f64, rng.below(300) as f64);
            src.push(p);
            dst.push(Vector2::new(rng.below(400) as f64, rng.below(300) as f64));
        }
        let params = RansacParams {
            inlier_threshold: 1e-6,
            min_inliers: 10,
            seed: 7,
            ..RansacParams::default()
        };
        let result = ransac_homography(&src, &dst, &params);
        let recovered = result.homography.expect("inlier consensus");
        assert!(
            result.inliers.len() >= 30,
            "found {} inliers",
            result.inliers.len()
        );
        for i in 0..3 {
            for j in 0..3 {
                let actual = recovered.normalized_for_output()[(i, j)];
                let expected = h.normalized_for_output()[(i, j)];
                assert!(
                    (actual - expected).abs() <= 1e-6,
                    "outlier-robust H[{i},{j}]: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn ransac_is_deterministic_for_a_seed() {
        let h = test_homography();
        let (mut src, mut dst) = inlier_correspondences(&h, 18, 3);
        src.push(Vector2::new(10.0, 10.0));
        dst.push(Vector2::new(-500.0, 400.0));
        let params = RansacParams {
            inlier_threshold: 1e-5,
            min_inliers: 6,
            seed: 12345,
            ..RansacParams::default()
        };
        let first = ransac_homography(&src, &dst, &params);
        let second = ransac_homography(&src, &dst, &params);
        assert_eq!(first.inliers, second.inliers);
        assert_eq!(first.iterations, second.iterations);
        let a = first.homography.expect("first run").normalized_for_output();
        let b = second
            .homography
            .expect("second run")
            .normalized_for_output();
        assert_eq!(a, b);
    }

    #[test]
    fn ransac_rejects_collinear_samples_d4() {
        let src: Vec<_> = (0..8)
            .map(|i| Vector2::new(i as f64, 2.0 * i as f64))
            .collect();
        let dst: Vec<_> = src
            .iter()
            .map(|p| Vector2::new(p.x + 1.0, p.y - 3.0))
            .collect();
        let params = RansacParams {
            max_iterations: 20,
            ..RansacParams::default()
        };
        let result = ransac_homography(&src, &dst, &params);
        assert!(result.homography.is_none());
        assert!(result.iterations <= params.max_iterations);
    }

    #[test]
    fn ransac_requires_four_correspondences_r1() {
        let src = vec![Vector2::new(0.0, 0.0); 3];
        let dst = vec![Vector2::new(1.0, 1.0); 3];
        let params = RansacParams {
            max_iterations: 20,
            ..RansacParams::default()
        };
        let result = ransac_homography(&src, &dst, &params);
        assert!(result.homography.is_none());
        assert!(result.iterations <= params.max_iterations);
        assert!(result.inliers.is_empty());
    }

    fn rectangle_scene() -> (
        Vec<crate::feature::EdgePoint>,
        crate::gradient::GradientField,
        crate::edge::DistanceField,
    ) {
        use crate::feature::{EdgePoint, EdgeSet};
        let width = 100;
        let height = 80;
        let (x0, x1, y0, y1) = (20, 80, 20, 60);
        let mut gx = vec![0.0f32; width * height];
        let mut gy = vec![0.0f32; width * height];
        let mut magnitude = vec![0.0f32; width * height];
        let mut edges = Vec::new();
        let mut points = Vec::new();
        let mut add = |x: f32, y: f32, nx: f32, ny: f32, sign: i8| {
            let strength = 0.3;
            // A blurred step produces a gradient band of a few pixels, not a
            // single-pixel spike.  Widening the test band matters for the
            // smoothness of the score landscape.
            for offset in -2i32..=2 {
                let sx = x + nx * offset as f32;
                let sy = y + ny * offset as f32;
                if sx < 0.0 || sy < 0.0 {
                    continue;
                }
                let (ix, iy) = (sx.round() as usize, sy.round() as usize);
                if ix >= width || iy >= height {
                    continue;
                }
                let weight = (-0.5 * (offset as f32 / 1.1).powi(2)).exp();
                let index = iy * width + ix;
                let value = strength * weight;
                if value > magnitude[index] {
                    magnitude[index] = value;
                    gx[index] = nx * value * sign as f32;
                    gy[index] = ny * value * sign as f32;
                }
            }
            edges.push(EdgePoint {
                x,
                y,
                nx,
                ny,
                magnitude: strength,
                sign,
                chain_id: 0,
                weight: 1.0,
            });
            points.push(EdgePoint {
                x,
                y,
                nx,
                ny,
                magnitude: 1.0,
                sign,
                chain_id: 0,
                weight: 1.0,
            });
        };
        for x in (x0 + 1..x1).step_by(3) {
            add(x as f32, y0 as f32, 0.0, 1.0, 1);
            add(x as f32, y1 as f32, 0.0, 1.0, -1);
        }
        for y in (y0 + 1..y1).step_by(3) {
            add(x0 as f32, y as f32, 1.0, 0.0, 1);
            add(x1 as f32, y as f32, 1.0, 0.0, -1);
        }
        let gradient = crate::gradient::GradientField {
            width,
            height,
            gx,
            gy,
            magnitude,
        };
        let edge_set = EdgeSet {
            points: edges,
            chains: Vec::new(),
        };
        let distance = crate::edge::DistanceField::from_edges(&edge_set, width, height);
        let model_points = points.into_iter().filter(|_| true).collect::<Vec<_>>();
        (model_points, gradient, distance)
    }

    #[test]
    fn icp_refines_a_nearby_similarity() {
        let (points, gradient, distance) = rectangle_scene();
        let pose = Similarity::from_pose(1.0, -0.8, 0.03, 1.03);
        let result = refine_similarity_level(
            &points,
            100,
            80,
            &gradient,
            &distance,
            pose,
            2.5,
            &crate::score::ScoreParams::default(),
            &SimilarityRefineParams::default(),
        );
        assert!(
            result.correspondences >= 8,
            "correspondences {}",
            result.correspondences
        );
        assert!(result.pose.tx.abs() < 0.35, "tx {}", result.pose.tx);
        assert!(result.pose.ty.abs() < 0.35, "ty {}", result.pose.ty);
        assert!(
            result.pose.theta.abs() < 0.01,
            "theta {}",
            result.pose.theta
        );
        assert!(
            (result.pose.scale - 1.0).abs() < 0.02,
            "scale {}",
            result.pose.scale
        );
    }

    #[test]
    fn pattern_search_recovers_from_a_coarse_seed() {
        let (points, gradient, distance) = rectangle_scene();
        let pose = Similarity::from_pose(1.5, -1.2, 0.05, 1.05);
        let refined = pattern_search_similarity(
            &points,
            100,
            80,
            &gradient,
            &distance,
            &pose,
            2.5,
            &crate::score::ScoreParams::default(),
            &PatternSearchParams {
                grid: true,
                ..PatternSearchParams::default()
            },
        );
        let score = crate::score::score_similarity(
            &points,
            100,
            80,
            &gradient,
            &distance,
            &refined,
            2.5,
            &crate::score::ScoreParams::default(),
        )
        .score;
        assert!(score > 0.85, "score {score}, pose {refined:?}");
    }

    #[test]
    fn degenerate_triple_is_detected() {
        let collinear = [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(3.0, 3.0),
            Vector2::new(0.0, 1.0),
        ];
        assert!(has_degenerate_triple(&collinear));
        let general = [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.0, 1.0),
            Vector2::new(2.0, 3.0),
        ];
        assert!(!has_degenerate_triple(&general));
    }

    fn tilted_scene(
        model_points: &[crate::feature::EdgePoint],
        homography: &Homography,
        width: usize,
        height: usize,
    ) -> (crate::gradient::GradientField, crate::edge::DistanceField) {
        use crate::feature::{EdgePoint, EdgeSet};
        let mut gx = vec![0.0f32; width * height];
        let mut gy = vec![0.0f32; width * height];
        let mut magnitude = vec![0.0f32; width * height];
        let mut edges = Vec::new();
        for point in model_points {
            let p = Vector2::new(f64::from(point.x), f64::from(point.y));
            let Some(q) = homography.try_act_point(p) else {
                continue;
            };
            let Some(normal) = homography
                .transform_normal(p, Vector2::new(f64::from(point.nx), f64::from(point.ny)))
            else {
                continue;
            };
            let strength = 0.3f32;
            for offset in -2i32..=2 {
                let sx = q.x as f32 + normal.x as f32 * offset as f32;
                let sy = q.y as f32 + normal.y as f32 * offset as f32;
                if sx < 0.0 || sy < 0.0 {
                    continue;
                }
                let (ix, iy) = (sx.round() as usize, sy.round() as usize);
                if ix >= width || iy >= height {
                    continue;
                }
                let weight = (-0.5 * (offset as f32 / 1.1).powi(2)).exp();
                let value = strength * weight;
                let index = iy * width + ix;
                if value > magnitude[index] {
                    magnitude[index] = value;
                    gx[index] = normal.x as f32 * value * f32::from(point.sign);
                    gy[index] = normal.y as f32 * value * f32::from(point.sign);
                }
            }
            edges.push(EdgePoint {
                x: q.x as f32,
                y: q.y as f32,
                nx: normal.x as f32,
                ny: normal.y as f32,
                magnitude: strength,
                sign: point.sign,
                chain_id: 0,
                weight: 1.0,
            });
        }
        let gradient = crate::gradient::GradientField {
            width,
            height,
            gx,
            gy,
            magnitude,
        };
        let distance = crate::edge::DistanceField::from_edges(
            &EdgeSet {
                points: edges,
                chains: Vec::new(),
            },
            width,
            height,
        );
        (gradient, distance)
    }

    fn perspective_truth() -> Homography {
        Homography::from_matrix(Matrix3::new(
            1.02, -0.03, 5.0, 0.04, 0.98, -3.0, 0.0005, -0.0003, 1.0,
        ))
    }

    fn rectangle_roi() -> [[f32; 2]; 4] {
        [[15.0, 15.0], [85.0, 15.0], [85.0, 65.0], [15.0, 65.0]]
    }

    #[test]
    fn correspondence_assembly_tracks_a_tilted_rectangle() {
        let (model_points, _, _) = rectangle_scene();
        let truth = perspective_truth();
        let (gradient, distance) = tilted_scene(&model_points, &truth, 200, 160);
        let input = HomographyInput {
            model_points: &model_points,
            model_corners: &[],
            model_segments: &[],
            roi_model: &rectangle_roi(),
            width: 200,
            height: 160,
            gradient: &gradient,
            distance: &distance,
            test_corners: &[],
            test_segments: &[],
            tau_d: 2.5,
            score_params: &crate::score::ScoreParams::default(),
        };
        let correspondences =
            assemble_correspondences(&input, &truth, &HomographyParams::default());
        assert!(
            correspondences.len() >= 16,
            "assembled {} correspondences",
            correspondences.len()
        );
        let inliers = count_correspondence_inliers(&truth, &correspondences, 0.75);
        assert!(
            inliers.len() >= correspondences.len() * 3 / 4,
            "{} of {} correspondences are inliers",
            inliers.len(),
            correspondences.len()
        );
    }

    #[test]
    fn ransac_recovers_a_homography_from_mixed_correspondences() {
        let truth = perspective_truth();
        let mut correspondences = Vec::new();
        let mut rng = SplitMix64::new(17);
        for _ in 0..40 {
            let p = Vector2::new(rng.below(120) as f64 + 10.0, rng.below(90) as f64 + 5.0);
            let Some(q) = truth.try_act_point(p) else {
                continue;
            };
            let normal = truth.transform_normal(p, Vector2::new(0.6, 0.8));
            correspondences.push(Correspondence {
                model: p,
                image: q,
                normal,
                weight: 1.0,
            });
        }
        for _ in 0..15 {
            correspondences.push(Correspondence {
                model: Vector2::new(rng.below(120) as f64, rng.below(90) as f64),
                image: Vector2::new(rng.below(200) as f64, rng.below(160) as f64),
                normal: None,
                weight: 1.0,
            });
        }
        let params = RansacParams {
            inlier_threshold: 0.5,
            min_inliers: 10,
            seed: 3,
            ..RansacParams::default()
        };
        let result = ransac_homography_correspondences(&correspondences, &params);
        let recovered = result.homography.expect("consensus homography");
        let expected = truth.normalized_for_output();
        let actual = recovered.normalized_for_output();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (actual[(i, j)] - expected[(i, j)]).abs() <= 1e-6,
                    "RANSAC mixed H[{i},{j}]: {} vs {}",
                    actual[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    #[test]
    fn refine_homography_accepts_perspective_evidence() {
        let (model_points, _, _) = rectangle_scene();
        let truth = perspective_truth();
        let (gradient, distance) = tilted_scene(&model_points, &truth, 200, 160);
        let local = truth.pose_at_origin().expect("truth pose");
        let pose = Similarity::from_pose(local.x, local.y, local.angle, local.scale);
        let score_params = crate::score::ScoreParams::default();
        let similarity_score = crate::score::score_similarity(
            &model_points,
            200,
            160,
            &gradient,
            &distance,
            &pose,
            2.5,
            &score_params,
        );
        let input = HomographyInput {
            model_points: &model_points,
            model_corners: &[],
            model_segments: &[],
            roi_model: &rectangle_roi(),
            width: 200,
            height: 160,
            gradient: &gradient,
            distance: &distance,
            test_corners: &[],
            test_segments: &[],
            tau_d: 2.5,
            score_params: &score_params,
        };
        let params = HomographyParams {
            min_similarity_score: 0.0,
            ransac: RansacParams {
                inlier_threshold: 1.0,
                min_inliers: 8,
                seed: 7,
                ..RansacParams::default()
            },
            ..HomographyParams::default()
        };
        let outcome = refine_homography(&input, pose, &similarity_score, &params);
        let accepted = outcome
            .homography
            .unwrap_or_else(|| panic!("fallback: {:?}", outcome.fallback_reason));
        assert!(
            outcome.score.score > similarity_score.score,
            "score {} did not improve on {}",
            outcome.score.score,
            similarity_score.score
        );
        assert!(outcome.condition.is_some());
        let mut worst = 0.0f64;
        for corner in rectangle_roi() {
            let p = Vector2::new(f64::from(corner[0]), f64::from(corner[1]));
            let expected = truth.try_act_point(p).expect("truth maps ROI");
            let actual = accepted.try_act_point(p).expect("refined maps ROI");
            worst = worst.max((actual - expected).norm());
        }
        assert!(
            worst <= 1.5,
            "ROI-corner reprojection {worst} px exceeds 1.5 px"
        );
    }

    #[test]
    fn fronto_parallel_scene_falls_back_to_similarity() {
        let (model_points, gradient, distance) = rectangle_scene();
        let pose = Similarity::IDENTITY;
        let score_params = crate::score::ScoreParams::default();
        let similarity_score = crate::score::score_similarity(
            &model_points,
            100,
            80,
            &gradient,
            &distance,
            &pose,
            2.5,
            &score_params,
        );
        let input = HomographyInput {
            model_points: &model_points,
            model_corners: &[],
            model_segments: &[],
            roi_model: &rectangle_roi(),
            width: 100,
            height: 80,
            gradient: &gradient,
            distance: &distance,
            test_corners: &[],
            test_segments: &[],
            tau_d: 2.5,
            score_params: &score_params,
        };
        let outcome = refine_homography(
            &input,
            pose,
            &similarity_score,
            &HomographyParams::default(),
        );
        assert!(outcome.homography.is_none());
        assert!(outcome.fallback_reason.is_some());
    }

    #[test]
    fn homography_disabled_reports_the_fallback() {
        let (model_points, gradient, distance) = rectangle_scene();
        let score_params = crate::score::ScoreParams::default();
        let pose = Similarity::IDENTITY;
        let score = crate::score::score_similarity(
            &model_points,
            100,
            80,
            &gradient,
            &distance,
            &pose,
            2.5,
            &score_params,
        );
        let input = HomographyInput {
            model_points: &model_points,
            model_corners: &[],
            model_segments: &[],
            roi_model: &rectangle_roi(),
            width: 100,
            height: 80,
            gradient: &gradient,
            distance: &distance,
            test_corners: &[],
            test_segments: &[],
            tau_d: 2.5,
            score_params: &score_params,
        };
        let params = HomographyParams {
            enabled: false,
            ..HomographyParams::default()
        };
        let outcome = refine_homography(&input, pose, &score, &params);
        assert!(outcome.homography.is_none());
        assert_eq!(
            outcome.fallback_reason,
            Some(HomographyFallbackReason::Disabled)
        );
        assert_eq!(outcome.correspondences, 0);
        assert_eq!(outcome.score, score);
    }

    #[test]
    fn validation_rejects_behind_camera_and_ill_conditioned() {
        let params = HomographyParams::default();
        let roi = [[10.0, -10.0], [10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0]];
        let behind =
            Homography::from_matrix(Matrix3::new(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01, 0.0, -1.0));
        assert_eq!(
            validate_homography(&behind, &roi, &params),
            Err(HomographyFallbackReason::DepthBehindCamera)
        );
        let ill_conditioned =
            Homography::from_matrix(Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e-9));
        assert_eq!(
            validate_homography(&ill_conditioned, &roi, &params),
            Err(HomographyFallbackReason::IllConditioned)
        );
        let singular =
            Homography::from_matrix(Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0));
        assert_eq!(
            validate_homography(&singular, &roi, &params),
            Err(HomographyFallbackReason::Degenerate)
        );
        let identity = Homography::IDENTITY;
        assert!(validate_homography(&identity, &roi, &params).is_ok());
    }

    #[test]
    fn insufficient_correspondences_report_the_fallback() {
        let params = HomographyParams {
            min_similarity_score: 0.0,
            ..HomographyParams::default()
        };
        let score = crate::score::ScoreDiagnostics::default();
        let input = HomographyInput {
            model_points: &[],
            model_corners: &[],
            model_segments: &[],
            roi_model: &rectangle_roi(),
            width: 100,
            height: 80,
            gradient: &crate::gradient::GradientField {
                width: 100,
                height: 80,
                gx: vec![0.0; 8000],
                gy: vec![0.0; 8000],
                magnitude: vec![0.0; 8000],
            },
            distance: &crate::edge::DistanceField::from_edges(
                &crate::feature::EdgeSet::default(),
                100,
                80,
            ),
            test_corners: &[],
            test_segments: &[],
            tau_d: 2.5,
            score_params: &crate::score::ScoreParams::default(),
        };
        let outcome = refine_homography(&input, Similarity::IDENTITY, &score, &params);
        assert!(outcome.homography.is_none());
        assert_eq!(
            outcome.fallback_reason,
            Some(HomographyFallbackReason::InsufficientCorrespondences)
        );
    }

    #[test]
    fn perspective_terms_that_move_across_rounds_are_unstable() {
        assert!(relative_perspective_change((1e-4, -2e-4), (1.1e-4, -1.9e-4)) < 0.35);
        assert!(relative_perspective_change((1e-4, -2e-4), (3e-4, -2e-4)) > 0.35);
        // Near-zero perspective is judged relative to the noise floor.
        assert!(relative_perspective_change((0.0, 0.0), (2e-6, 0.0)) > 0.35);
    }
}
