//! Correspondence refinement and RANSAC homography estimation.
//!
//! M4 provides the shared RANSAC machinery: deterministic sampling from a
//! seeded PRNG, degeneracy rejection, adaptive iteration counts, inlier
//! counting by symmetric transfer error, and refitting on the inlier set.  The
//! similarity ICP and robust Levenberg–Marquardt stages of README §9.6–§9.7
//! build on this in M5–M6.
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

use crate::geometry::{Homography, normalized_dlt};

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
