//! Deterministic synthetic benchmark harness (README §13.3).
//!
//! The harness renders a taught model into generated scenes with known
//! similarity poses, occlusion, clutter, contrast changes, border clipping, and
//! plane tilt.  It reports recall, false detections, pose error, homography
//! acceptance, and ROI-corner reprojection.  All randomness comes from
//! [`SplitMix64`], seeded per case, so a run is reproducible across platforms.
//!
//! [`SyntheticRenderer`] is shared by `pm-cli bench synthetic` and the
//! integration benchmark.  The integration test pins its historical scene
//! radius and foreground gate so milestone-to-milestone regression numbers stay
//! comparable; the CLI uses the model-derived defaults.

use nalgebra::{Matrix3, Vector2};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::geometry::{Homography, Similarity};
use crate::gray::GrayF32;
use crate::model::{HypothesisSources, Model};
use crate::score::Detection;
use crate::search::{SearchParams, detect, model_roi_in_model_coords};

/// Synthetic-benchmark report format identifier.
pub const BENCHMARK_FORMAT: &str = "pattern-match-benchmark";
/// Current synthetic-benchmark report version.
pub const BENCHMARK_VERSION: u32 = 1;

const SUPERSAMPLE: usize = 6;
const MIN_SCENE_SIDE: usize = 128;
const MAX_SCENE_SIDE: usize = 1024;
const SCENE_SEED_STRIDE: u64 = 7919;

/// A small deterministic PRNG (SplitMix64).
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform value in `[low, high)`.
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }
}

/// Parameters for [`run_benchmark`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyntheticBenchParams {
    /// Number of clean-similarity cases.  The other scenarios use a fraction of
    /// this count.
    pub cases: usize,
    /// Base deterministic seed.
    pub seed: u64,
}

impl Default for SyntheticBenchParams {
    fn default() -> Self {
        Self {
            cases: 200,
            seed: 42,
        }
    }
}

/// Renders a taught model into deterministic synthetic scenes.
///
/// Scene dimensions, background level, and the model radius are derived from
/// the model so the harness works for any taught shape, not only the L-bracket
/// used by the integration benchmark.  The model's embedded source image is
/// required.
pub struct SyntheticRenderer<'a> {
    source: &'a GrayF32,
    roi: crate::geometry::RotatedRect,
    zero: crate::geometry::ZeroFrame,
    width: usize,
    height: usize,
    background: f32,
    extent: f32,
    foreground_threshold: f32,
    rotation_min: f32,
    rotation_max: f32,
    scale_min: f32,
    scale_max: f32,
}

impl<'a> SyntheticRenderer<'a> {
    /// Build a renderer from a model with an embedded source image.
    pub fn new(model: &'a Model) -> Result<Self> {
        let source = &model.source.as_ref().ok_or_else(|| {
            Error::InvalidParameter(
                "model has no embedded source image; rebuild it with the source enabled".to_owned(),
            )
        })?;
        if !model.roi.is_valid() || source.gray.is_empty() {
            return Err(Error::InvalidParameter(
                "model ROI or embedded source is invalid".to_owned(),
            ));
        }
        let extent = model
            .features
            .edges
            .points
            .iter()
            .map(|point| point.x.hypot(point.y))
            .fold(0.0f32, f32::max);
        if !extent.is_finite() || extent <= 0.0 {
            return Err(Error::InvalidParameter(
                "model has no edge extent to synthesize from".to_owned(),
            ));
        }
        let fit = (extent * model.search_defaults.scale_max.max(1.0) + 3.0) * 2.0;
        let width = ((model.roi.width * 2.0).round() as usize)
            .max(fit as usize)
            .clamp(MIN_SCENE_SIDE, MAX_SCENE_SIDE);
        let height = ((model.roi.height * 1.75).round() as usize)
            .max(fit as usize)
            .clamp(MIN_SCENE_SIDE, MAX_SCENE_SIDE);
        Ok(Self {
            source: &source.gray,
            roi: model.roi,
            zero: model.zero,
            width,
            height,
            background: estimate_background(&source.gray),
            extent,
            foreground_threshold: 0.0,
            rotation_min: model.search_defaults.rotation_min_deg.to_radians(),
            rotation_max: model.search_defaults.rotation_max_deg.to_radians(),
            scale_min: model.search_defaults.scale_min,
            scale_max: model.search_defaults.scale_max,
        })
    }

    /// Scene dimensions in pixels.
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Override the model radius used for pose margins and scene fitting.
    ///
    /// The default is the largest taught edge radius; callers that want to
    /// reproduce a historical benchmark may pin it explicitly.
    #[must_use]
    pub fn with_extent(mut self, extent: f32) -> Self {
        if extent.is_finite() && extent > 0.0 {
            self.extent = extent;
            let fit = (self.extent * self.scale_max.max(1.0) + 3.0) * 2.0;
            self.width = self
                .width
                .max(fit as usize)
                .clamp(MIN_SCENE_SIDE, MAX_SCENE_SIDE);
            self.height = self
                .height
                .max(fit as usize)
                .clamp(MIN_SCENE_SIDE, MAX_SCENE_SIDE);
        }
        self
    }

    /// Set the intensity difference from the scene background required before a
    /// pixel receives model appearance.
    ///
    /// The default of zero composites the whole ROI.  A positive threshold
    /// leaves near-background pixels untouched, which avoids pasting a faint
    /// rectangle when the source background is not perfectly flat.
    #[must_use]
    pub fn with_foreground_threshold(mut self, threshold: f32) -> Self {
        if threshold.is_finite() && threshold >= 0.0 {
            self.foreground_threshold = threshold;
        }
        self
    }

    /// Background level estimated from the source border.
    #[must_use]
    pub fn background(&self) -> f32 {
        self.background
    }

    /// Maximum model-frame radius of the taught edge points.
    #[must_use]
    pub fn extent(&self) -> f32 {
        self.extent
    }

    /// A random similarity pose that keeps the model inside the scene.
    pub fn random_pose(&self, rng: &mut SplitMix64, border_clip: bool) -> Similarity {
        let theta = rng.range(self.rotation_min, self.rotation_max);
        let scale = rng.range(self.scale_min, self.scale_max);
        let extent = self.extent * scale;
        let margin = extent + 3.0;
        let (mut tx, mut ty) = self.centered_translation(rng, margin);
        if border_clip {
            let side = (rng.next_u64() % 4) as usize;
            match side {
                0 => tx = rng.range(extent * 0.25, extent * 0.75),
                1 => tx = self.width as f32 - rng.range(extent * 0.25, extent * 0.75),
                2 => ty = rng.range(extent * 0.25, extent * 0.75),
                _ => ty = self.height as f32 - rng.range(extent * 0.25, extent * 0.75),
            }
        }
        Similarity::from_pose(
            f64::from(tx),
            f64::from(ty),
            f64::from(theta),
            f64::from(scale),
        )
    }

    /// A similarity base pose for a tilted scene.
    pub fn random_tilt_base(&self, rng: &mut SplitMix64) -> Similarity {
        let theta = rng.range(self.rotation_min, self.rotation_max);
        let scale = rng.range(0.7, 1.6);
        let margin = self.extent * 1.25 * scale + 3.0;
        let (tx, ty) = self.centered_translation(rng, margin);
        Similarity::from_pose(
            f64::from(tx),
            f64::from(ty),
            f64::from(theta),
            f64::from(scale),
        )
    }

    fn centered_translation(&self, rng: &mut SplitMix64, margin: f32) -> (f32, f32) {
        let width = self.width as f32;
        let height = self.height as f32;
        let x = if width > 2.0 * margin {
            rng.range(margin, width - margin)
        } else {
            rng.range(width * 0.25, width * 0.75)
        };
        let y = if height > 2.0 * margin {
            rng.range(margin, height - margin)
        } else {
            rng.range(height * 0.25, height * 0.75)
        };
        (x, y)
    }

    /// Render similarity poses into a scene.
    ///
    /// Pixels are supersampled with a box filter so edges stay anti-aliased;
    /// only points inside the taught ROI contribute model appearance.
    #[must_use]
    pub fn render(&self, poses: &[Similarity], contrast: f32, brightness: f32) -> GrayF32 {
        let mut data = vec![self.background; self.width * self.height];
        for pose in poses {
            let Some(inverse) = pose.try_inverse() else {
                continue;
            };
            let bounds = self.pose_bounds(pose);
            for y in bounds.1..=bounds.3 {
                for x in bounds.0..=bounds.2 {
                    let mut sum = 0.0f32;
                    let mut covered = 0usize;
                    let mut foreground = false;
                    for sy in 0..SUPERSAMPLE {
                        for sx in 0..SUPERSAMPLE {
                            let qx = x as f32 + (sx as f32 - 2.5) / SUPERSAMPLE as f32;
                            let qy = y as f32 + (sy as f32 - 2.5) / SUPERSAMPLE as f32;
                            let p_model =
                                inverse.transform_point(Vector2::new(qx as f64, qy as f64));
                            let p_image = self.zero.to_image([p_model.x as f32, p_model.y as f32]);
                            if self.roi.contains(p_image[0], p_image[1]) {
                                let value = self.source.sample_bilinear(p_image[0], p_image[1]);
                                sum += value;
                                covered += 1;
                                foreground |=
                                    (value - self.background).abs() > self.foreground_threshold;
                            }
                        }
                    }
                    if covered > 0 && foreground {
                        data[y * self.width + x] = sum / covered as f32;
                    }
                }
            }
        }
        apply_appearance(&mut data, contrast, brightness);
        GrayF32::new(self.width, self.height, data).expect("scene dimensions are non-zero")
    }

    /// Render a truth homography (plane tilt) into a scene.
    #[must_use]
    pub fn render_tilted(&self, truth: &Homography) -> GrayF32 {
        let mut data = vec![self.background; self.width * self.height];
        let Some(inverse) = truth.try_inverse() else {
            return GrayF32::new(self.width, self.height, data)
                .expect("scene dimensions are non-zero");
        };
        let bounds = self.homography_bounds(truth);
        let samples = 4usize;
        for y in bounds.1..=bounds.3 {
            for x in bounds.0..=bounds.2 {
                let mut sum = 0.0f32;
                let mut covered = 0usize;
                let mut foreground = false;
                for sy in 0..samples {
                    for sx in 0..samples {
                        let qx = x as f32 + (sx as f32 + 0.5) / samples as f32 - 0.5;
                        let qy = y as f32 + (sy as f32 + 0.5) / samples as f32 - 0.5;
                        let Some(p_model) =
                            inverse.try_act_point(Vector2::new(f64::from(qx), f64::from(qy)))
                        else {
                            continue;
                        };
                        let p_image = self.zero.to_image([p_model.x as f32, p_model.y as f32]);
                        if self.roi.contains(p_image[0], p_image[1]) {
                            let value = self.source.sample_bilinear(p_image[0], p_image[1]);
                            sum += value;
                            covered += 1;
                            foreground |=
                                (value - self.background).abs() > self.foreground_threshold;
                        }
                    }
                }
                if covered > 0 && foreground {
                    data[y * self.width + x] = sum / covered as f32;
                }
            }
        }
        GrayF32::new(self.width, self.height, data).expect("scene dimensions are non-zero")
    }

    fn pose_bounds(&self, pose: &Similarity) -> (usize, usize, usize, usize) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for corner in self.roi.corners() {
            let corner_model = self.zero.to_model(corner);
            let q = pose.transform_point(Vector2::new(
                f64::from(corner_model[0]),
                f64::from(corner_model[1]),
            ));
            min_x = min_x.min(q.x as f32);
            min_y = min_y.min(q.y as f32);
            max_x = max_x.max(q.x as f32);
            max_y = max_y.max(q.y as f32);
        }
        self.clamped_bounds(min_x, min_y, max_x, max_y)
    }

    fn homography_bounds(&self, truth: &Homography) -> (usize, usize, usize, usize) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for corner in self.roi.corners() {
            let corner_model = self.zero.to_model(corner);
            let p = Vector2::new(f64::from(corner_model[0]), f64::from(corner_model[1]));
            if let Some(q) = truth.try_act_point(p) {
                min_x = min_x.min(q.x as f32);
                min_y = min_y.min(q.y as f32);
                max_x = max_x.max(q.x as f32);
                max_y = max_y.max(q.y as f32);
            }
        }
        if !min_x.is_finite() {
            return (0, 0, self.width - 1, self.height - 1);
        }
        self.clamped_bounds(min_x, min_y, max_x, max_y)
    }

    fn clamped_bounds(
        &self,
        min_x: f32,
        min_y: f32,
        max_x: f32,
        max_y: f32,
    ) -> (usize, usize, usize, usize) {
        let x0 = (min_x.floor() as isize - 2).clamp(0, self.width as isize - 1) as usize;
        let y0 = (min_y.floor() as isize - 2).clamp(0, self.height as isize - 1) as usize;
        let x1 = (max_x.ceil() as isize + 2).clamp(0, self.width as isize - 1) as usize;
        let y1 = (max_y.ceil() as isize + 2).clamp(0, self.height as isize - 1) as usize;
        (x0, y0, x1.max(x0), y1.max(y0))
    }

    /// A physically derived plane tilt about the model origin.
    ///
    /// The camera looks down `z`; the plane basis is `R_x(tilt_x) R_y(tilt_y)`
    /// applied to the model axes and translated to distance `d`.  The focal
    /// length is chosen so the local scale at the model origin equals the base
    /// similarity scale.  Both tilt components are at most the requested
    /// magnitude.
    #[must_use]
    pub fn tilt_homography(
        &self,
        rng: &mut SplitMix64,
        base: &Similarity,
        min_deg: f32,
        max_deg: f32,
    ) -> Homography {
        let magnitude = rng.range(min_deg, max_deg).to_radians();
        let azimuth = rng.range(0.0, std::f32::consts::TAU);
        let tilt_x = magnitude * azimuth.sin();
        let tilt_y = magnitude * azimuth.cos();
        let distance = rng.range(500.0, 800.0);
        let cos_x = tilt_x.cos();
        let cos_y = tilt_y.cos();
        let focal = distance / (cos_x * cos_y).sqrt().max(1e-3);
        let tilt = Matrix3::new(
            f64::from(focal * cos_y),
            0.0,
            0.0,
            f64::from(focal * tilt_x.sin() * tilt_y.sin()),
            f64::from(focal * cos_x),
            0.0,
            f64::from(-cos_x * tilt_y.sin()),
            f64::from(tilt_x.sin()),
            f64::from(distance),
        );
        base.embed().compose(&Homography::from_matrix(tilt))
    }

    /// Cover roughly 30% of the visible object with a rectangular patch.
    pub fn occlude(&self, image: &mut GrayF32, pose: &Similarity, rng: &mut SplitMix64) {
        let Some(inverse) = pose.try_inverse() else {
            return;
        };
        let mut object_pixels = Vec::new();
        for y in 0..self.height {
            for x in 0..self.width {
                let p_model = inverse.transform_point(Vector2::new(x as f64, y as f64));
                let p_image = self.zero.to_image([p_model.x as f32, p_model.y as f32]);
                if !self.roi.contains(p_image[0], p_image[1]) {
                    continue;
                }
                let value = self.source.sample_bilinear(p_image[0], p_image[1]);
                if (value - self.background).abs() > 0.2 {
                    object_pixels.push((x, y));
                }
            }
        }
        if object_pixels.is_empty() {
            return;
        }
        let target = (object_pixels.len() as f32 * 0.3) as usize;
        let aspect = rng.range(0.6, 1.6);
        let width = ((target as f32 * aspect).sqrt() as usize).max(4);
        let height = ((target as f32 / aspect).sqrt() as usize).max(4);
        let (center_x, center_y) = object_pixels[rng.next_u64() as usize % object_pixels.len()];
        let x0 = center_x.saturating_sub(width / 2);
        let y0 = center_y.saturating_sub(height / 2);
        for y in y0..(y0 + height).min(self.height) {
            for x in x0..(x0 + width).min(self.width) {
                image.data[y * self.width + x] = self.background;
            }
        }
    }

    /// Draw procedural clutter strokes.
    pub fn add_clutter(&self, image: &mut GrayF32, rng: &mut SplitMix64, count: usize) {
        for _ in 0..count {
            let x0 = rng.range(0.0, self.width as f32);
            let y0 = rng.range(0.0, self.height as f32);
            let length = rng.range(8.0, 45.0);
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let value = rng.range(0.18, 0.35);
            let thickness = if rng.unit() < 0.5 { 1 } else { 2 };
            let dx = angle.cos();
            let dy = angle.sin();
            for step in 0..length as i32 {
                for offset in -(thickness / 2)..=(thickness / 2) {
                    let x = (x0 + step as f32 * dx - offset as f32 * dy).round() as i32;
                    let y = (y0 + step as f32 * dy + offset as f32 * dx).round() as i32;
                    if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                        image.data[y as usize * self.width + x as usize] = value;
                    }
                }
            }
        }
    }
}

/// Whether a detection matches a truth pose using the benchmark tolerances.
#[must_use]
pub fn detection_matches(detection: &Detection, truth: &Similarity) -> bool {
    let distance = (detection.pose.tx - truth.tx).hypot(detection.pose.ty - truth.ty);
    let angle = (detection.pose.theta - truth.theta).rem_euclid(std::f64::consts::TAU);
    let angle = angle.min(std::f64::consts::TAU - angle);
    let scale_ratio = (detection.pose.scale / truth.scale).max(truth.scale / detection.pose.scale);
    distance < 4.0 && angle < 10.0f64.to_radians() && scale_ratio < 1.1
}

/// Maximum ROI-corner reprojection error of a reported homography, in pixels.
#[must_use]
pub fn roi_reprojection_error(
    detection: &Detection,
    truth: &Homography,
    roi_model: &[[f32; 2]; 4],
) -> f32 {
    let detected = Homography::from_array(detection.homography);
    roi_model
        .iter()
        .map(|corner| {
            let p = Vector2::new(f64::from(corner[0]), f64::from(corner[1]));
            match (truth.try_act_point(p), detected.try_act_point(p)) {
                (Some(expected), Some(actual)) => (expected - actual).norm() as f32,
                _ => f32::INFINITY,
            }
        })
        .fold(0.0f32, f32::max)
}

/// Accumulated benchmark metrics for one scenario.
#[derive(Clone, Debug, Default)]
pub struct BenchmarkMetrics {
    /// Number of scenes evaluated.
    pub cases: usize,
    /// Number of synthetic objects.
    pub ground_truth: usize,
    /// Objects with a matching detection.
    pub true_positives: usize,
    /// Detections not matched to an object.
    pub false_positives: usize,
    /// Correct detections with an accepted homography.
    pub homography_accepted: usize,
    /// Correct detections that fell back to the similarity embedding.
    pub homography_fallback: usize,
    /// Fallback detections that did not report a reason.
    pub fallback_reason_missing: usize,
    position_errors: Vec<f32>,
    angle_errors: Vec<f32>,
    scale_errors: Vec<f32>,
    worst_reprojection: Option<f32>,
    worst_fallback_reprojection: Option<f32>,
}

impl BenchmarkMetrics {
    /// Add a similarity scene with known truth poses.
    pub fn add_similarity(&mut self, truths: &[Similarity], detections: &[Detection]) {
        self.associate(truths, detections);
    }

    /// Add a tilted scene, tracking ROI-corner reprojection error.
    pub fn add_tilt(
        &mut self,
        truth_pose: &Similarity,
        truth_homography: &Homography,
        roi_model: &[[f32; 2]; 4],
        detections: &[Detection],
    ) {
        for index in self.associate(std::slice::from_ref(truth_pose), detections) {
            let detection = &detections[index];
            let error = roi_reprojection_error(detection, truth_homography, roi_model);
            if detection.homography_estimated {
                self.worst_reprojection = Some(
                    self.worst_reprojection
                        .map_or(error, |worst| worst.max(error)),
                );
            } else {
                self.worst_fallback_reprojection = Some(
                    self.worst_fallback_reprojection
                        .map_or(error, |worst| worst.max(error)),
                );
            }
        }
    }

    fn associate(&mut self, truths: &[Similarity], detections: &[Detection]) -> Vec<usize> {
        self.cases += 1;
        self.ground_truth += truths.len();
        let mut used = vec![false; detections.len()];
        let mut matched = Vec::with_capacity(truths.len());
        for truth in truths {
            if let Some((index, detection)) = detections
                .iter()
                .enumerate()
                .filter(|(index, _)| !used[*index])
                .find(|(_, detection)| detection_matches(detection, truth))
            {
                used[index] = true;
                matched.push(index);
                self.record_match(detection, truth);
            }
        }
        self.false_positives += used.iter().filter(|used| !**used).count();
        matched
    }

    fn record_match(&mut self, detection: &Detection, truth: &Similarity) {
        self.true_positives += 1;
        if detection.homography_estimated {
            self.homography_accepted += 1;
        } else {
            self.homography_fallback += 1;
            if detection.diagnostics.fallback_reason.is_none() {
                self.fallback_reason_missing += 1;
            }
        }
        self.position_errors
            .push(((detection.pose.tx - truth.tx).hypot(detection.pose.ty - truth.ty)) as f32);
        let angle = (detection.pose.theta - truth.theta).rem_euclid(std::f64::consts::TAU);
        let angle = angle.min(std::f64::consts::TAU - angle);
        self.angle_errors.push(angle.to_degrees() as f32);
        self.scale_errors
            .push(((detection.pose.scale / truth.scale - 1.0).abs() * 100.0) as f32);
    }

    /// Fraction of synthetic objects that produced a matching detection.
    #[must_use]
    pub fn recall(&self) -> f32 {
        if self.ground_truth == 0 {
            1.0
        } else {
            self.true_positives as f32 / self.ground_truth as f32
        }
    }

    /// False detections per scene.
    #[must_use]
    pub fn false_positives_per_image(&self) -> f32 {
        self.false_positives as f32 / self.cases.max(1) as f32
    }

    /// Largest zero-position error among matching detections.
    #[must_use]
    pub fn max_position_error(&self) -> f32 {
        max_or_zero(&self.position_errors)
    }

    /// Largest angle error among matching detections.
    #[must_use]
    pub fn max_angle_error(&self) -> f32 {
        max_or_zero(&self.angle_errors)
    }

    /// Largest scale error among matching detections, in percent.
    #[must_use]
    pub fn max_scale_error(&self) -> f32 {
        max_or_zero(&self.scale_errors)
    }

    /// Convert the accumulated metrics into a serializable report row.
    #[must_use]
    pub fn report(&self, name: impl Into<String>) -> ScenarioReport {
        ScenarioReport {
            name: name.into(),
            cases: self.cases,
            ground_truth: self.ground_truth,
            true_positives: self.true_positives,
            false_positives: self.false_positives,
            recall: self.recall(),
            false_positives_per_image: self.false_positives_per_image(),
            max_position_error_px: self.max_position_error(),
            max_angle_error_deg: self.max_angle_error(),
            max_scale_error_percent: self.max_scale_error(),
            homography_accepted: self.homography_accepted,
            homography_fallback: self.homography_fallback,
            homography_accepted_fraction: self.homography_accepted as f32
                / self.true_positives.max(1) as f32,
            fallback_reason_missing: self.fallback_reason_missing,
            max_roi_reprojection_px: self.worst_reprojection,
            max_fallback_roi_reprojection_px: self.worst_fallback_reprojection,
        }
    }
}

fn max_or_zero(values: &[f32]) -> f32 {
    values.iter().copied().fold(0.0, f32::max)
}

/// One scenario in a [`BenchmarkReport`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioReport {
    /// Scenario name.
    pub name: String,
    /// Scenes evaluated.
    pub cases: usize,
    /// Synthetic objects in those scenes.
    pub ground_truth: usize,
    /// Objects with a matching detection.
    pub true_positives: usize,
    /// Detections without a matching object.
    pub false_positives: usize,
    /// `true_positives / ground_truth`.
    pub recall: f32,
    /// `false_positives / cases`.
    pub false_positives_per_image: f32,
    /// Largest zero-position error among matching detections, pixels.
    pub max_position_error_px: f32,
    /// Largest angle error among matching detections, degrees.
    pub max_angle_error_deg: f32,
    /// Largest scale error among matching detections, percent.
    pub max_scale_error_percent: f32,
    /// Correct detections with an accepted homography.
    pub homography_accepted: usize,
    /// Correct detections that fell back to the similarity embedding.
    pub homography_fallback: usize,
    /// `homography_accepted / true_positives`.
    pub homography_accepted_fraction: f32,
    /// Fallback detections with no reason recorded.
    pub fallback_reason_missing: usize,
    /// Largest ROI-corner reprojection error for accepted homographies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_roi_reprojection_px: Option<f32>,
    /// Largest ROI-corner reprojection error for fallback detections.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_fallback_roi_reprojection_px: Option<f32>,
}

/// A complete synthetic benchmark report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Always [`BENCHMARK_FORMAT`].
    pub format: String,
    /// Major report version.
    pub version: u32,
    /// Model name.
    pub model: String,
    /// Base seed used for scene generation.
    pub seed: u64,
    /// Requested clean-scene case count.
    pub cases: usize,
    /// Per-scenario metrics.
    pub scenarios: Vec<ScenarioReport>,
}

impl BenchmarkReport {
    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|error| Error::Format(error.to_string()))
    }
}

/// Run the full deterministic synthetic benchmark for `model`.
///
/// Uses the model's stored search defaults and requires an embedded source
/// image.
pub fn run_benchmark(model: &Model, params: &SyntheticBenchParams) -> Result<BenchmarkReport> {
    run_benchmark_with_params(model, params, &SearchParams::from(model.search_defaults))
}

/// Run the synthetic benchmark with explicit search parameters.
///
/// Requires a model with an embedded source image.
pub fn run_benchmark_with_params(
    model: &Model,
    params: &SyntheticBenchParams,
    search: &SearchParams,
) -> Result<BenchmarkReport> {
    let renderer = SyntheticRenderer::new(model)?;
    let search = *search;
    let hough = SearchParams {
        hypothesis_sources: HypothesisSources::Hough,
        ..search
    };
    let cases = params.cases.max(1);
    let ancillary = (cases / 4).max(1);
    let mut scenarios = Vec::new();

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &search,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 1000,
            cases,
            name: "clean_similarity",
        },
        |renderer, rng| {
            let truth = renderer.random_pose(rng, false);
            let scene = renderer.render(&[truth], rng.range(0.9, 1.1), rng.range(-0.03, 0.03));
            (scene, vec![truth])
        },
    )?);

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &hough,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 1000,
            cases: ancillary,
            name: "clean_similarity_hough_only",
        },
        |renderer, rng| {
            let truth = renderer.random_pose(rng, false);
            let scene = renderer.render(&[truth], 1.0, 0.0);
            (scene, vec![truth])
        },
    )?);

    for (name, source_params) in [
        ("low_contrast_hybrid", &search),
        ("low_contrast_hough_only", &hough),
    ] {
        scenarios.push(run_similarity_scenario(
            model,
            &renderer,
            source_params,
            ScenarioSpec {
                seed: params.seed,
                seed_offset: 1500,
                cases: ancillary,
                name,
            },
            |renderer, rng| {
                let truth = renderer.random_pose(rng, false);
                let scene = renderer.render(&[truth], 0.5, 0.0);
                (scene, vec![truth])
            },
        )?);
    }

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &search,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 2000,
            cases: ancillary,
            name: "occlusion_30pct",
        },
        |renderer, rng| {
            let truth = renderer.random_pose(rng, false);
            let mut scene = renderer.render(&[truth], 1.0, 0.0);
            renderer.occlude(&mut scene, &truth, rng);
            (scene, vec![truth])
        },
    )?);

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &search,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 3000,
            cases: ancillary,
            name: "clutter_only",
        },
        |renderer, rng| {
            let (width, height) = renderer.dimensions();
            let mut scene = GrayF32::filled(width, height, renderer.background());
            renderer.add_clutter(&mut scene, rng, 25);
            (scene, Vec::new())
        },
    )?);

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &search,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 4000,
            cases: ancillary,
            name: "clutter_with_object",
        },
        |renderer, rng| {
            let truth = renderer.random_pose(rng, false);
            let mut scene = renderer.render(&[truth], 1.0, 0.0);
            renderer.add_clutter(&mut scene, rng, 20);
            (scene, vec![truth])
        },
    )?);

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &search,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 5000,
            cases: ancillary,
            name: "border_clipped",
        },
        |renderer, rng| {
            let truth = renderer.random_pose(rng, true);
            let scene = renderer.render(&[truth], 1.0, 0.0);
            (scene, vec![truth])
        },
    )?);

    scenarios.push(run_similarity_scenario(
        model,
        &renderer,
        &search,
        ScenarioSpec {
            seed: params.seed,
            seed_offset: 6000,
            cases: ancillary,
            name: "multi_instance",
        },
        |renderer, rng| {
            let first = renderer.random_pose(rng, false);
            let separation = renderer.extent() * 1.7;
            let mut second = renderer.random_pose(rng, false);
            let mut attempts = 0;
            while (first.tx - second.tx).abs() < f64::from(separation)
                && (first.ty - second.ty).abs() < f64::from(separation)
                && attempts < 32
            {
                second = renderer.random_pose(rng, false);
                attempts += 1;
            }
            let scene = renderer.render(&[first, second], 1.0, 0.0);
            (scene, vec![first, second])
        },
    )?);

    scenarios.push(run_tilt_scenario(
        model,
        &renderer,
        &search,
        params.seed,
        ancillary,
    )?);

    Ok(BenchmarkReport {
        format: BENCHMARK_FORMAT.to_owned(),
        version: BENCHMARK_VERSION,
        model: model.name.clone(),
        seed: params.seed,
        cases,
        scenarios,
    })
}

fn scene_seed(base: u64, offset: u64, index: usize) -> u64 {
    base.wrapping_add(offset)
        .wrapping_add(index as u64 * SCENE_SEED_STRIDE)
}

struct ScenarioSpec<'a> {
    seed: u64,
    seed_offset: u64,
    cases: usize,
    name: &'a str,
}

fn run_similarity_scenario(
    model: &Model,
    renderer: &SyntheticRenderer<'_>,
    params: &SearchParams,
    spec: ScenarioSpec<'_>,
    mut make_scene: impl FnMut(&SyntheticRenderer<'_>, &mut SplitMix64) -> (GrayF32, Vec<Similarity>),
) -> Result<ScenarioReport> {
    let mut metrics = BenchmarkMetrics::default();
    for index in 0..spec.cases {
        let mut rng = SplitMix64::new(scene_seed(spec.seed, spec.seed_offset, index));
        let (scene, truths) = make_scene(renderer, &mut rng);
        let result = detect(model, &scene, params)?;
        metrics.add_similarity(&truths, &result.detections);
    }
    Ok(metrics.report(spec.name))
}

fn run_tilt_scenario(
    model: &Model,
    renderer: &SyntheticRenderer<'_>,
    params: &SearchParams,
    seed: u64,
    cases: usize,
) -> Result<ScenarioReport> {
    let roi_model = model_roi_in_model_coords(model);
    let mut metrics = BenchmarkMetrics::default();
    for index in 0..cases {
        let mut rng = SplitMix64::new(scene_seed(seed, 9000, index));
        let base = renderer.random_tilt_base(&mut rng);
        let truth = renderer.tilt_homography(&mut rng, &base, 8.0, 20.0);
        let local = truth
            .pose_at_origin()
            .ok_or_else(|| Error::InvalidParameter("tilt truth is singular".to_owned()))?;
        let truth_pose = Similarity::from_pose(local.x, local.y, local.angle, local.scale);
        let scene = renderer.render_tilted(&truth);
        let result = detect(model, &scene, params)?;
        metrics.add_tilt(&truth_pose, &truth, &roi_model, &result.detections);
    }
    Ok(metrics.report("tilt_20deg"))
}

fn estimate_background(source: &GrayF32) -> f32 {
    let mut values = Vec::with_capacity(2 * (source.width + source.height));
    let last_x = source.width - 1;
    let last_y = source.height - 1;
    for x in 0..source.width {
        values.push(source.data[x]);
        values.push(source.data[last_y * source.width + x]);
    }
    for y in 0..source.height {
        values.push(source.data[y * source.width]);
        values.push(source.data[y * source.width + last_x]);
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn apply_appearance(data: &mut [f32], contrast: f32, brightness: f32) {
    for value in data {
        *value = ((*value - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_mix_is_deterministic_and_bounded() {
        let mut first = SplitMix64::new(7);
        let mut second = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(first.next_u64(), second.next_u64());
        }
        for _ in 0..100 {
            let value = first.unit();
            assert!((0.0..1.0).contains(&value));
        }
        for _ in 0..100 {
            let value = second.range(-3.0, 5.0);
            assert!((-3.0..5.0).contains(&value));
        }
    }

    #[test]
    fn background_estimation_uses_border() {
        let mut data = vec![0.7f32; 16 * 16];
        for y in 4..12 {
            for x in 4..12 {
                data[y * 16 + x] = 0.1;
            }
        }
        let image = GrayF32::new(16, 16, data).unwrap();
        assert!((estimate_background(&image) - 0.7).abs() < 1e-6);
    }
}
