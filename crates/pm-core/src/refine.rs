//! Correspondence refinement and RANSAC homography estimation.
//!
//! M4 provides the shared RANSAC machinery: deterministic sampling from a
//! seeded PRNG, degeneracy rejection, adaptive iteration counts, inlier
//! counting by symmetric transfer error, and refitting on the inlier set.
//!
//! M5 adds the normal-directed correspondence search and weighted similarity
//! Procrustes fit of README §9.6, with a score-based pattern-search fallback
//! and a fine final polish.  Robust Levenberg–Marquardt homography refinement
//! (README §9.7) arrives with M6.
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

use nalgebra::Vector2;

use crate::geometry::{Homography, Similarity, normalized_dlt};

/// Minimal sample size for a homography (four correspondences).
pub const HOMOGRAPHY_SAMPLE_SIZE: usize = 4;

/// RANSAC configuration (README §9.7, Appendix A).
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// Parameters for similarity refinement (README §9.6).
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
}
