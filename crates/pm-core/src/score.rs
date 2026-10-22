//! Edge-based scoring, confidence, and pose non-maximum suppression
//! (README §9.5, §9.8, §9.9).
//!
//! Scoring projects model edge points through a similarity pose or a
//! homography, samples the test-level distance transform and gradient field,
//! and accumulates a normalized weighted quality.  The homography path maps
//! each normal by the inverse-transpose of the local Jacobian; both paths are
//! normalization-compatible so the perspective validation gate can compare
//! their scores directly.

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

use crate::edge::DistanceField;
use crate::feature::EdgePoint;
use crate::geometry::{Homography, NUMERIC_EPS, Similarity, homogenize};
use crate::gradient::GradientField;

/// Which coarse stage produced a hypothesis (README §9.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedSource {
    /// Oriented generalized Hough voting.
    #[default]
    Hough,
    /// Corner-descriptor match.
    CornerDescriptor,
    /// Segment-pair invariant match.
    SegmentPair,
}

impl SeedSource {
    /// Stable diagnostic name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Hough => "hough",
            Self::CornerDescriptor => "corner_descriptor",
            Self::SegmentPair => "segment_pair",
        }
    }
}

/// Scoring parameters (README §9.5, Appendix A).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreParams {
    /// Distance tolerance in full-resolution pixels; scaled by the level.
    pub tau_d: f32,
    /// Orientation-agreement exponent.
    pub orientation_power: f32,
    /// Gradient magnitude below which reliability is zero.
    pub reliability_min: f32,
    /// Gradient magnitude at which reliability reaches one.
    pub reliability_max: f32,
    /// Minimum visible model weight for a usable detection.
    pub coverage_min: f32,
    /// Whether contrast polarity must agree.
    pub polarity_enabled: bool,
}

impl Default for ScoreParams {
    fn default() -> Self {
        Self {
            tau_d: 2.5,
            orientation_power: 2.0,
            reliability_min: 0.01,
            reliability_max: 0.06,
            coverage_min: 0.4,
            polarity_enabled: true,
        }
    }
}

/// Normalized scoring diagnostics for one pose.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScoreDiagnostics {
    /// Weighted match fraction in `[0, 1]`.
    pub score: f32,
    /// Visible model weight divided by all model weight.
    pub coverage: f32,
    /// Model weight with positive quality divided by visible weight.
    pub matched_fraction: f32,
    /// Weighted mean distance of matched points, in level pixels.
    pub mean_distance: f32,
    /// Weighted mean orientation agreement in `[0, 1]`.
    pub mean_orientation_agreement: f32,
}

/// Score a similarity pose against one test level.
///
/// `model_points` and `pose` are expressed in level-local coordinates; `tau_d`
/// is the level-scaled distance tolerance.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn score_similarity(
    model_points: &[EdgePoint],
    width: usize,
    height: usize,
    gradient: &GradientField,
    distance: &DistanceField,
    pose: &Similarity,
    tau_d: f32,
    params: &ScoreParams,
) -> ScoreDiagnostics {
    if model_points.is_empty() || width == 0 || height == 0 || tau_d <= 0.0 {
        return ScoreDiagnostics::default();
    }
    let rotation = Similarity::rotation(pose.theta);
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;
    let power = params.orientation_power.max(0.0) as f64;
    let mut total_weight = 0.0f64;
    let mut visible_weight = 0.0f64;
    let mut matched_weight = 0.0f64;
    let mut quality_sum = 0.0f64;
    let mut distance_sum = 0.0f64;
    let mut orientation_sum = 0.0f64;

    for point in model_points {
        let weight = f64::from(point.weight.max(0.0));
        total_weight += weight;
        let q = pose.transform_point(Vector2::new(f64::from(point.x), f64::from(point.y)));
        if q.x < 0.0 || q.y < 0.0 || q.x > max_x || q.y > max_y {
            continue;
        }
        visible_weight += weight;

        let normal = rotation * Vector2::new(f64::from(point.nx), f64::from(point.ny));
        let (gx, gy) = gradient.sample(q.x as f32, q.y as f32);
        let (gx, gy) = (f64::from(gx), f64::from(gy));
        let magnitude = gx.hypot(gy);
        let reliability = reliability(magnitude, params);
        let agreement = if magnitude > 1e-9 {
            let dot = (normal.x * gx + normal.y * gy) / magnitude;
            if params.polarity_enabled {
                (dot * f64::from(point.sign)).max(0.0)
            } else {
                dot.abs()
            }
        } else {
            0.0
        };
        let edge_distance = f64::from(distance.sample(q.x as f32, q.y as f32));
        let proximity = (1.0 - edge_distance / f64::from(tau_d)).max(0.0);
        let quality = weight * proximity * orientation_power_value(agreement, power) * reliability;
        quality_sum += quality;
        orientation_sum += weight * agreement * reliability;
        if quality > 0.0 {
            matched_weight += weight;
            distance_sum += weight * edge_distance;
        }
    }

    finish_score(
        total_weight,
        visible_weight,
        matched_weight,
        quality_sum,
        distance_sum,
        orientation_sum,
    )
}

/// `agreement^power`, with the common integer exponents handled directly.
fn orientation_power_value(agreement: f64, power: f64) -> f64 {
    if power == 2.0 {
        agreement * agreement
    } else if power == 1.0 {
        agreement
    } else if power == 0.0 {
        1.0
    } else {
        agreement.powf(power)
    }
}

/// Score a homography against one test level (README §9.5, §9.7).
///
/// Each model point is projected through `homography`; its normal is mapped by
/// the inverse-transpose of the local Jacobian.  Points behind the camera
/// (non-positive homogeneous depth) are treated as invisible.  The score is the
/// same normalized weighted fraction as [`score_similarity`], so a similarity
/// fallback and an estimated homography can be compared directly by the
/// validation gate.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn score_homography(
    model_points: &[EdgePoint],
    width: usize,
    height: usize,
    gradient: &GradientField,
    distance: &DistanceField,
    homography: &Homography,
    tau_d: f32,
    params: &ScoreParams,
) -> ScoreDiagnostics {
    if model_points.is_empty() || width == 0 || height == 0 || tau_d <= 0.0 {
        return ScoreDiagnostics::default();
    }
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;
    let power = params.orientation_power.max(0.0) as f64;
    let mut total_weight = 0.0f64;
    let mut visible_weight = 0.0f64;
    let mut matched_weight = 0.0f64;
    let mut quality_sum = 0.0f64;
    let mut distance_sum = 0.0f64;
    let mut orientation_sum = 0.0f64;

    for point in model_points {
        let weight = f64::from(point.weight.max(0.0));
        total_weight += weight;
        let p = Vector2::new(f64::from(point.x), f64::from(point.y));
        let mapped = homography.act_homogeneous(homogenize(p));
        if !mapped.z.is_finite() || mapped.z <= NUMERIC_EPS {
            continue;
        }
        let q = Vector2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if q.x < 0.0 || q.y < 0.0 || q.x > max_x || q.y > max_y {
            continue;
        }
        visible_weight += weight;

        let normal =
            homography.transform_normal(p, Vector2::new(f64::from(point.nx), f64::from(point.ny)));
        let (gx, gy) = gradient.sample(q.x as f32, q.y as f32);
        let (gx, gy) = (f64::from(gx), f64::from(gy));
        let magnitude = gx.hypot(gy);
        let reliability = reliability(magnitude, params);
        let agreement = match normal {
            Some(normal) if magnitude > 1e-9 => {
                let dot = (normal.x * gx + normal.y * gy) / magnitude;
                if params.polarity_enabled {
                    (dot * f64::from(point.sign)).max(0.0)
                } else {
                    dot.abs()
                }
            }
            _ => 0.0,
        };
        let edge_distance = f64::from(distance.sample(q.x as f32, q.y as f32));
        let proximity = (1.0 - edge_distance / f64::from(tau_d)).max(0.0);
        let quality = weight * proximity * orientation_power_value(agreement, power) * reliability;
        quality_sum += quality;
        orientation_sum += weight * agreement * reliability;
        if quality > 0.0 {
            matched_weight += weight;
            distance_sum += weight * edge_distance;
        }
    }

    finish_score(
        total_weight,
        visible_weight,
        matched_weight,
        quality_sum,
        distance_sum,
        orientation_sum,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_score(
    total_weight: f64,
    visible_weight: f64,
    matched_weight: f64,
    quality_sum: f64,
    distance_sum: f64,
    orientation_sum: f64,
) -> ScoreDiagnostics {
    let score = if visible_weight > 0.0 {
        (quality_sum / visible_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let coverage = if total_weight > 0.0 {
        (visible_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let matched_fraction = if visible_weight > 0.0 {
        (matched_weight / visible_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mean_distance = if matched_weight > 0.0 {
        (distance_sum / matched_weight) as f32
    } else {
        0.0
    };
    let mean_orientation_agreement = if visible_weight > 0.0 {
        (orientation_sum / visible_weight).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    ScoreDiagnostics {
        score: score as f32,
        coverage: coverage as f32,
        matched_fraction: matched_fraction as f32,
        mean_distance,
        mean_orientation_agreement,
    }
}

fn reliability(magnitude: f64, params: &ScoreParams) -> f64 {
    let low = f64::from(params.reliability_min.max(0.0));
    let high = f64::from(params.reliability_max.max(params.reliability_min));
    if magnitude <= low {
        0.0
    } else if magnitude >= high || high <= low + 1e-9 {
        1.0
    } else {
        let t = (magnitude - low) / (high - low);
        t * t * (3.0 - 2.0 * t)
    }
}

/// Per-detection diagnostics (README §9.9).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DetectionDiagnostics {
    /// Visible model weight fraction.
    pub coverage: f32,
    /// Matched weight fraction of visible weight.
    pub matched_fraction: f32,
    /// Mean edge distance of matched points, full-resolution pixels.
    pub mean_distance_px: f32,
    /// Mean orientation agreement.
    pub mean_orientation_agreement: f32,
    /// Coarse hypothesis source.
    pub seed_source: SeedSource,
    /// Number of correspondences assembled for homography estimation, or zero
    /// when the homography stage did not run.
    pub correspondences: usize,
    /// Correspondences consistent with the final reported homography.
    pub inliers: usize,
    /// RANSAC iterations executed for the homography estimate.
    pub ransac_iterations: usize,
    /// Corner correspondences assembled under the final geometry.  Kept out of
    /// the result JSON so the documented schema stays exact; used for tuning
    /// and diagnostic tests.
    #[serde(skip)]
    pub corner_correspondences: usize,
    /// Corner correspondences consistent with the final homography.
    #[serde(skip)]
    pub corner_inliers: usize,
    /// Homography condition number, when estimated.
    pub homography_condition: Option<f64>,
    /// Margin to the best overlapping competing detection.
    pub ambiguity: f32,
    /// Similarity-fallback reason, when applicable.
    pub fallback_reason: Option<String>,
}

/// A single detection.
///
/// This is the internal record.  For the versioned result JSON of README
/// §12.3, serialize a [`crate::result::ResultsFile`]; the derived serde form
/// here is an internal representation and does not rename pose fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Sequential detection id, assigned by NMS.
    pub id: usize,
    /// Reported pose at the model zero.  For an accepted homography this is the
    /// local Jacobian decomposition of README §4, not a global similarity.
    pub pose: Similarity,
    /// 3×3 homography mapping model to image.  When
    /// [`Detection::homography_estimated`] is false this embeds the similarity.
    pub homography: [[f64; 3]; 3],
    /// Whether the homography was estimated from perspective evidence.
    pub homography_estimated: bool,
    /// Weighted edge score in `[0, 1]`.
    pub score: f32,
    /// Heuristic confidence in `[0, 1]`.
    pub confidence: f32,
    /// Auditable diagnostics.
    pub diagnostics: DetectionDiagnostics,
}

/// NMS thresholds (README §9.8, Appendix A).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NmsParams {
    /// Zero-distance fraction of the model diagonal.
    pub distance_fraction: f32,
    /// Angle threshold in radians.
    pub angle_threshold: f32,
    /// Scale ratio allowance.
    pub scale_ratio: f32,
    /// Transformed-ROI IoU threshold.
    pub iou_threshold: f32,
}

impl Default for NmsParams {
    fn default() -> Self {
        Self {
            distance_fraction: 0.35,
            angle_threshold: 10.0f32.to_radians(),
            scale_ratio: 0.15,
            iou_threshold: 0.45,
        }
    }
}

/// Heuristic confidence from the README §9.9 formula.
#[must_use]
pub fn confidence_from(
    score: f32,
    matched_fraction: f32,
    mean_orientation_agreement: f32,
    ambiguity: f32,
    homography_estimated: bool,
) -> f32 {
    let base = 0.55 * score.clamp(0.0, 1.0)
        + 0.20 * matched_fraction.clamp(0.0, 1.0)
        + 0.15 * mean_orientation_agreement.clamp(0.0, 1.0)
        + 0.10 * ambiguity.clamp(0.0, 1.0);
    let perspective = if homography_estimated { 1.0 } else { 0.85 };
    (base * perspective).clamp(0.0, 1.0)
}

/// Apply multi-instance non-maximum suppression and assign sequential ids.
///
/// `roi_model` holds the taught ROI corners in model coordinates.
#[must_use]
pub fn non_maximum_suppression(
    mut detections: Vec<Detection>,
    model_diagonal: f32,
    roi_model: &[[f32; 2]; 4],
    max_detections: usize,
    params: &NmsParams,
) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.pose.tx.total_cmp(&b.pose.tx))
            .then_with(|| a.pose.ty.total_cmp(&b.pose.ty))
            .then_with(|| a.id.cmp(&b.id))
    });
    let all = detections.clone();
    let mut accepted: Vec<Detection> = Vec::new();
    for detection in detections {
        if accepted.len() >= max_detections {
            break;
        }
        let suppressed = accepted
            .iter()
            .any(|kept| overlaps(kept, &detection, model_diagonal, roi_model, params));
        if !suppressed {
            accepted.push(detection);
        }
    }

    for (index, detection) in accepted.iter_mut().enumerate() {
        let competitor = all
            .iter()
            .filter(|other| {
                other.id != detection.id
                    && overlaps(detection, other, model_diagonal, roi_model, params)
            })
            .map(|other| other.score)
            .fold(0.0f32, f32::max);
        detection.diagnostics.ambiguity = if competitor > 0.0 {
            ((detection.score - competitor) / detection.score.max(1e-3)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        detection.confidence = confidence_from(
            detection.score,
            detection.diagnostics.matched_fraction,
            detection.diagnostics.mean_orientation_agreement,
            detection.diagnostics.ambiguity,
            detection.homography_estimated,
        );
        detection.id = index;
    }
    accepted
}

/// Transformed ROI corners for a similarity pose.
#[must_use]
pub fn transformed_roi(pose: &Similarity, roi_model: &[[f32; 2]; 4]) -> [[f64; 2]; 4] {
    roi_model.map(|corner| {
        let q = pose.transform_point(Vector2::new(f64::from(corner[0]), f64::from(corner[1])));
        [q.x, q.y]
    })
}

fn overlaps(
    a: &Detection,
    b: &Detection,
    model_diagonal: f32,
    roi_model: &[[f32; 2]; 4],
    params: &NmsParams,
) -> bool {
    let distance = (a.pose.tx - b.pose.tx).hypot(a.pose.ty - b.pose.ty);
    let angle = angle_distance(a.pose.theta as f32, b.pose.theta as f32);
    let scale_limit = 1.0 + params.scale_ratio;
    let ratio = (a.pose.scale / b.pose.scale)
        .max(b.pose.scale / a.pose.scale)
        .abs();
    let scale_max = a.pose.scale.max(b.pose.scale) as f32;
    let same_pose = distance
        < f64::from(params.distance_fraction * model_diagonal * scale_max.max(1e-3))
        && angle < params.angle_threshold
        && ratio < f64::from(scale_limit);
    if same_pose {
        return true;
    }
    let first = detection_roi(a, roi_model);
    let second = detection_roi(b, roi_model);
    polygon_iou(&first, &second) > f64::from(params.iou_threshold)
}

/// Transformed ROI corners for a detection: the homography when one was
/// estimated, otherwise the similarity embedding.
fn detection_roi(detection: &Detection, roi_model: &[[f32; 2]; 4]) -> [[f64; 2]; 4] {
    if !detection.homography_estimated {
        return transformed_roi(&detection.pose, roi_model);
    }
    let homography = Homography::from_array(detection.homography);
    roi_model.map(|corner| {
        let p = Vector2::new(f64::from(corner[0]), f64::from(corner[1]));
        homography
            .try_act_point(p)
            .map_or([p.x, p.y], |mapped| [mapped.x, mapped.y])
    })
}

fn angle_distance(a: f32, b: f32) -> f32 {
    ((a - b + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI).abs()
}

/// Area of a convex polygon.
#[must_use]
pub fn polygon_area(polygon: &[[f64; 2]]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area.abs() * 0.5
}

/// Intersection-over-union of two convex polygons.
#[must_use]
pub fn polygon_iou(first: &[[f64; 2]; 4], second: &[[f64; 2]; 4]) -> f64 {
    let first_area = polygon_area(first);
    let second_area = polygon_area(second);
    if first_area <= 1e-12 || second_area <= 1e-12 {
        return 0.0;
    }
    let clipped = clip_convex(first, second);
    let intersection = polygon_area(&clipped);
    intersection / (first_area + second_area - intersection).max(1e-12)
}

fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut clip = clip.to_vec();
    if signed_area(&clip) < 0.0 {
        clip.reverse();
    }
    let mut output = subject.to_vec();
    for index in 0..clip.len() {
        if output.is_empty() {
            break;
        }
        let a = clip[index];
        let b = clip[(index + 1) % clip.len()];
        let input = std::mem::take(&mut output);
        let mut previous = input[input.len() - 1];
        let mut inside_previous = cross(
            b[0] - a[0],
            b[1] - a[1],
            previous[0] - a[0],
            previous[1] - a[1],
        ) >= 0.0;
        for &current in &input {
            let inside_current = cross(
                b[0] - a[0],
                b[1] - a[1],
                current[0] - a[0],
                current[1] - a[1],
            ) >= 0.0;
            if inside_current {
                if !inside_previous {
                    output.push(line_intersection(previous, current, a, b));
                }
                output.push(current);
            } else if inside_previous {
                output.push(line_intersection(previous, current, a, b));
            }
            previous = current;
            inside_previous = inside_current;
        }
    }
    output
}

fn signed_area(polygon: &[[f64; 2]]) -> f64 {
    let mut area = 0.0;
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area * 0.5
}

fn cross(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ax * by - ay * bx
}

fn line_intersection(p1: [f64; 2], p2: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let r = [p2[0] - p1[0], p2[1] - p1[1]];
    let s = [b[0] - a[0], b[1] - a[1]];
    let denominator = cross(r[0], r[1], s[0], s[1]);
    if denominator.abs() < 1e-12 {
        return p2;
    }
    let t = cross(a[0] - p1[0], a[1] - p1[1], s[0], s[1]) / denominator;
    [p1[0] + t * r[0], p1[1] + t * r[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::EdgeSet;
    use crate::geometry::matrix_to_array;
    use nalgebra::Matrix3;

    fn edge_point(x: f32, y: f32, nx: f32, ny: f32) -> EdgePoint {
        EdgePoint {
            x,
            y,
            nx,
            ny,
            magnitude: 1.0,
            sign: 1,
            chain_id: 0,
            weight: 1.0,
        }
    }

    fn horizontal_edge_field() -> (GradientField, DistanceField) {
        let width = 40;
        let height = 40;
        let mut magnitude = vec![0.0f32; width * height];
        let mut gy = vec![0.0f32; width * height];
        for x in 0..width {
            magnitude[20 * width + x] = 0.2;
            gy[20 * width + x] = 0.2;
        }
        let gradient = GradientField {
            width,
            height,
            gx: vec![0.0; width * height],
            gy,
            magnitude,
        };
        let edges = EdgeSet {
            points: (0..width)
                .map(|x| edge_point(x as f32, 20.0, 0.0, 1.0))
                .collect(),
            chains: Vec::new(),
        };
        let distance = DistanceField::from_edges(&edges, width, height);
        (gradient, distance)
    }

    #[test]
    fn score_is_maximal_for_an_exact_edge_pose() {
        let (gradient, distance) = horizontal_edge_field();
        let points = vec![
            edge_point(10.0, 0.0, 0.0, 1.0),
            edge_point(20.0, 0.0, 0.0, 1.0),
            edge_point(30.0, 0.0, 0.0, 1.0),
        ];
        let pose = Similarity::from_pose(0.0, 20.0, 0.0, 1.0);
        let result = score_similarity(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &pose,
            2.5,
            &ScoreParams::default(),
        );
        assert!(result.score > 0.99, "score {}", result.score);
        assert!(result.coverage > 0.99);
        assert!((result.mean_distance).abs() < 1e-3);
    }

    #[test]
    fn score_drops_with_distance_and_misalignment() {
        let (gradient, distance) = horizontal_edge_field();
        let points = vec![edge_point(20.0, 0.0, 0.0, 1.0)];
        let exact = score_similarity(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &Similarity::from_pose(0.0, 20.0, 0.0, 1.0),
            2.5,
            &ScoreParams::default(),
        );
        let shifted = score_similarity(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &Similarity::from_pose(0.0, 23.0, 0.0, 1.0),
            2.5,
            &ScoreParams::default(),
        );
        let rotated = score_similarity(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &Similarity::from_pose(0.0, 20.0, std::f64::consts::FRAC_PI_2, 1.0),
            2.5,
            &ScoreParams::default(),
        );
        assert!(
            exact.score > shifted.score,
            "{} vs {}",
            exact.score,
            shifted.score
        );
        assert!(
            exact.score > rotated.score,
            "{} vs {}",
            exact.score,
            rotated.score
        );
    }

    #[test]
    fn homography_score_matches_similarity_for_an_embedded_pose() {
        let (gradient, distance) = horizontal_edge_field();
        let points = vec![
            edge_point(10.0, 0.0, 0.0, 1.0),
            edge_point(20.0, 0.0, 0.0, 1.0),
            edge_point(30.0, 0.0, 0.0, 1.0),
        ];
        let pose = Similarity::from_pose(2.0, 19.0, 0.05, 1.02);
        let similarity = score_similarity(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &pose,
            2.5,
            &ScoreParams::default(),
        );
        let homography = score_homography(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &pose.embed(),
            2.5,
            &ScoreParams::default(),
        );
        assert!(
            (similarity.score - homography.score).abs() < 1e-5,
            "similarity {} vs homography {}",
            similarity.score,
            homography.score
        );
        assert!(
            (similarity.coverage - homography.coverage).abs() < 1e-5,
            "coverage differs"
        );
    }

    #[test]
    fn homography_score_is_maximal_for_an_exact_translation() {
        let (gradient, distance) = horizontal_edge_field();
        let points = vec![
            edge_point(10.0, 0.0, 0.0, 1.0),
            edge_point(20.0, 0.0, 0.0, 1.0),
            edge_point(30.0, 0.0, 0.0, 1.0),
        ];
        let homography =
            Homography::from_matrix(Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0));
        let result = score_homography(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &homography,
            2.5,
            &ScoreParams::default(),
        );
        assert!(result.score > 0.99, "score {}", result.score);
        assert!((result.mean_distance).abs() < 1e-3);
    }

    #[test]
    fn points_behind_the_camera_are_invisible() {
        let (gradient, distance) = horizontal_edge_field();
        let points = vec![
            edge_point(10.0, 0.0, 0.0, 1.0),
            edge_point(20.0, 0.0, 0.0, 1.0),
        ];
        let behind =
            Homography::from_matrix(Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01, 0.0, -1.0));
        let result = score_homography(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &behind,
            2.5,
            &ScoreParams::default(),
        );
        assert_eq!(result.coverage, 0.0);
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn coverage_excludes_points_outside_the_image() {
        let (gradient, distance) = horizontal_edge_field();
        let points = vec![
            edge_point(0.0, 0.0, 0.0, 1.0),
            edge_point(0.0, 0.0, 0.0, 1.0),
            edge_point(0.0, 0.0, 0.0, 1.0),
            edge_point(0.0, 0.0, 0.0, 1.0),
        ];
        let pose = Similarity::from_pose(100.0, 100.0, 0.0, 1.0);
        let result = score_similarity(
            &points,
            40,
            40,
            &gradient,
            &distance,
            &pose,
            2.5,
            &ScoreParams::default(),
        );
        assert_eq!(result.coverage, 0.0);
        assert_eq!(result.score, 0.0);
    }

    fn detection(x: f64, y: f64, angle: f64, scale: f64, score: f32) -> Detection {
        let pose = Similarity::from_pose(x, y, angle, scale);
        Detection {
            id: 0,
            pose,
            homography: matrix_to_array(&pose.embed().normalized_for_output()),
            homography_estimated: false,
            score,
            confidence: 0.0,
            diagnostics: DetectionDiagnostics {
                seed_source: SeedSource::Hough,
                ..DetectionDiagnostics::default()
            },
        }
    }

    #[test]
    fn nms_keeps_distinct_instances_and_suppresses_duplicates() {
        let roi = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]];
        let mut first = detection(50.0, 50.0, 0.1, 1.0, 0.9);
        first.id = 0;
        let mut duplicate = detection(51.0, 50.5, 0.12, 1.02, 0.85);
        duplicate.id = 1;
        let mut second = detection(150.0, 50.0, 0.1, 1.0, 0.8);
        second.id = 2;
        let result = non_maximum_suppression(
            vec![first, duplicate, second],
            80.0,
            &roi,
            10,
            &NmsParams::default(),
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 0);
        assert_eq!(result[1].id, 1);
        assert!(result[0].diagnostics.ambiguity < 0.2);
        assert!(result[1].diagnostics.ambiguity > 0.9);
    }

    #[test]
    fn polygon_iou_handles_overlapping_squares() {
        let a = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let b = [[5.0, 0.0], [15.0, 0.0], [15.0, 10.0], [5.0, 10.0]];
        let iou = polygon_iou(&a, &b);
        assert!((iou - 1.0 / 3.0).abs() < 1e-9, "iou {iou}");
    }

    #[test]
    fn similarity_embedding_matches_pose() {
        let pose = Similarity::from_pose(12.0, -3.0, 0.4, 1.3);
        let matrix = pose.embed().normalized_for_output();
        let expected = Matrix3::new(
            1.3 * 0.4f64.cos(),
            -1.3 * 0.4f64.sin(),
            12.0,
            1.3 * 0.4f64.sin(),
            1.3 * 0.4f64.cos(),
            -3.0,
            0.0,
            0.0,
            1.0,
        );
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[(i, j)] - expected[(i, j)]).abs() < 1e-12);
            }
        }
    }
}
