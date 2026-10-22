//! Rotation/scale-aware corner shape-context descriptors and segment-pair
//! invariants for hypothesis seeding (README §9.3).
//!
//! Descriptors are computed in model coordinates relative to each corner's
//! dominant orientation.  Three descriptor radii are concatenated so later
//! matching can compare a radius ratio (scale change) as well as shape.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::feature::{Corner, EdgeSet, Segment};

/// Radial bins per shape-context scale.
pub const RADIAL_BINS: usize = 5;
/// Angular bins per shape-context scale.
pub const ANGULAR_BINS: usize = 12;
/// Gradient-orientation bins per shape-context scale.
pub const ORIENTATION_BINS: usize = 8;
/// Descriptor radii in model pixels (taught scale).
pub const DESCRIPTOR_RADII: [f32; 3] = [12.0, 24.0, 48.0];
/// Floats per descriptor scale: shape context plus orientation histogram.
pub const DESCRIPTOR_SCALE_DIM: usize = RADIAL_BINS * ANGULAR_BINS + ORIENTATION_BINS;
/// Floats per corner descriptor (all scales concatenated).
pub const DESCRIPTOR_DIM: usize = DESCRIPTOR_SCALE_DIM * DESCRIPTOR_RADII.len();

/// Shape-context descriptor for one model corner.
#[derive(Clone, Debug, PartialEq)]
pub struct CornerDescriptor {
    /// Index into the model's corner array.
    pub corner_index: usize,
    /// `DESCRIPTOR_DIM` L1-normalized floats.
    pub values: Vec<f32>,
}

/// Compute descriptors for model corners using model edge points.
#[must_use]
pub fn describe_corners(corners: &[Corner], edges: &EdgeSet) -> Vec<CornerDescriptor> {
    corners
        .par_iter()
        .enumerate()
        .map(|(corner_index, corner)| CornerDescriptor {
            corner_index,
            values: describe_corner(corner, edges),
        })
        .collect()
}

fn describe_corner(corner: &Corner, edges: &EdgeSet) -> Vec<f32> {
    let mut values = Vec::with_capacity(DESCRIPTOR_DIM);
    let (sin, cos) = corner.orientation.sin_cos();
    for radius in DESCRIPTOR_RADII {
        let mut bins = vec![0.0f32; DESCRIPTOR_SCALE_DIM];
        for point in &edges.points {
            let dx = point.x - corner.x;
            let dy = point.y - corner.y;
            let distance = dx.hypot(dy);
            if distance <= 1e-6 || distance > radius {
                continue;
            }
            let weight = point.weight.max(0.0);
            // Rotate the offset into the corner's frame.
            let rx = cos * dx + sin * dy;
            let ry = -sin * dx + cos * dy;
            let angle = ry.atan2(rx).rem_euclid(std::f32::consts::TAU);
            let radial = radial_bin(distance, radius);
            let angular = ((angle / std::f32::consts::TAU) * ANGULAR_BINS as f32)
                .floor()
                .clamp(0.0, (ANGULAR_BINS - 1) as f32) as usize;
            bins[radial * ANGULAR_BINS + angular] += weight;

            let normal_angle = point.ny.atan2(point.nx);
            let relative = (normal_angle - corner.orientation).rem_euclid(std::f32::consts::PI);
            let orientation = ((relative / std::f32::consts::PI) * ORIENTATION_BINS as f32)
                .floor()
                .clamp(0.0, (ORIENTATION_BINS - 1) as f32) as usize;
            bins[RADIAL_BINS * ANGULAR_BINS + orientation] += weight;
        }
        l1_normalize(&mut bins);
        values.extend_from_slice(&bins);
    }
    values
}

fn radial_bin(distance: f32, radius: f32) -> usize {
    const INNER_RADIUS: f32 = 2.0;
    if distance <= INNER_RADIUS || radius <= INNER_RADIUS {
        return 0;
    }
    let t = (distance / INNER_RADIUS).ln() / (radius / INNER_RADIUS).ln();
    (t * RADIAL_BINS as f32)
        .floor()
        .clamp(0.0, (RADIAL_BINS - 1) as f32) as usize
}

fn l1_normalize(values: &mut [f32]) {
    let sum: f32 = values.iter().map(|v| v.abs()).sum();
    if sum > 0.0 {
        for value in values {
            *value /= sum;
        }
    }
}

/// Similarity-invariant descriptors of a pair of non-parallel segments.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentPairInvariant {
    /// Index of the first segment.
    pub first: usize,
    /// Index of the second segment.
    pub second: usize,
    /// `min(length) / max(length)`.
    pub length_ratio: f32,
    /// Angle between the undirected directions in `[0, π/2]`.
    pub angle_between: f32,
    /// Midpoint distance normalized by the combined length.
    pub midpoint_distance: f32,
    /// Angle of the midpoint vector relative to the first segment direction.
    pub midpoint_angle: f32,
}

/// Build segment-pair invariants, skipping near-parallel pairs.
///
/// Pairs are emitted in deterministic index order and capped at `max_pairs`.
#[must_use]
pub fn segment_pair_invariants(
    segments: &[Segment],
    max_pairs: usize,
) -> Vec<SegmentPairInvariant> {
    let mut pairs = Vec::new();
    for first in 0..segments.len() {
        for second in first + 1..segments.len() {
            if pairs.len() >= max_pairs {
                return pairs;
            }
            let a = &segments[first];
            let b = &segments[second];
            if a.length <= 1e-6 || b.length <= 1e-6 {
                continue;
            }
            let da = ((a.x1 - a.x0) / a.length, (a.y1 - a.y0) / a.length);
            let db = ((b.x1 - b.x0) / b.length, (b.y1 - b.y0) / b.length);
            let dot = (da.0 * db.0 + da.1 * db.1).abs().clamp(0.0, 1.0);
            let angle_between = dot.acos();
            if angle_between < 0.087 {
                continue;
            }
            let ma = ((a.x0 + a.x1) * 0.5, (a.y0 + a.y1) * 0.5);
            let mb = ((b.x0 + b.x1) * 0.5, (b.y0 + b.y1) * 0.5);
            let dm = (mb.0 - ma.0, mb.1 - ma.1);
            let midpoint_distance = dm.0.hypot(dm.1) / (a.length + b.length);
            let midpoint_angle = (da.0 * dm.1 - da.1 * dm.0)
                .abs()
                .atan2(da.0 * dm.0 + da.1 * dm.1);

            pairs.push(SegmentPairInvariant {
                first,
                second,
                length_ratio: a.length.min(b.length) / a.length.max(b.length),
                angle_between,
                midpoint_distance,
                midpoint_angle,
            });
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{EdgeChain, EdgePoint};

    fn synthetic_features() -> (Vec<Corner>, EdgeSet) {
        let corner = Corner {
            x: 20.0,
            y: 15.0,
            response: 1.0,
            orientation: 0.35,
            weight: 1.0,
        };
        let mut points = Vec::new();
        for i in 0..24 {
            let angle = i as f32 * std::f32::consts::TAU / 24.0;
            let radius = 5.0 + (i % 5) as f32 * 3.0;
            points.push(EdgePoint {
                x: corner.x + radius * angle.cos(),
                y: corner.y + radius * angle.sin(),
                nx: angle.cos(),
                ny: angle.sin(),
                magnitude: 1.0,
                sign: 1,
                chain_id: i as u32,
                weight: 0.5 + 0.02 * i as f32,
            });
        }
        let chain_indices: Vec<usize> = (0..points.len()).collect();
        (
            vec![corner],
            EdgeSet {
                points,
                chains: vec![EdgeChain {
                    id: 0,
                    points: chain_indices,
                }],
            },
        )
    }

    fn rotate_features(corners: &mut [Corner], edges: &mut EdgeSet, angle: f32) {
        let (sin, cos) = angle.sin_cos();
        for corner in corners.iter_mut() {
            let (x, y) = (corner.x, corner.y);
            corner.x = cos * x - sin * y;
            corner.y = sin * x + cos * y;
            corner.orientation = (corner.orientation + angle).rem_euclid(std::f32::consts::PI);
        }
        for point in &mut edges.points {
            let (x, y) = (point.x, point.y);
            point.x = cos * x - sin * y;
            point.y = sin * x + cos * y;
            let (nx, ny) = (point.nx, point.ny);
            point.nx = cos * nx - sin * ny;
            point.ny = sin * nx + cos * ny;
        }
    }

    #[test]
    fn descriptor_has_expected_dimensions_and_norms() {
        let (corners, edges) = synthetic_features();
        let descriptors = describe_corners(&corners, &edges);
        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].values.len(), DESCRIPTOR_DIM);
        for scale in descriptors[0].values.as_chunks::<DESCRIPTOR_SCALE_DIM>().0 {
            let sum: f32 = scale.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "scale sum {sum}");
        }
    }

    #[test]
    fn descriptor_is_rotation_invariant() {
        let (corners, edges) = synthetic_features();
        let base = describe_corners(&corners, &edges);
        let (mut rotated_corners, mut rotated_edges) = (corners.clone(), edges.clone());
        rotate_features(&mut rotated_corners, &mut rotated_edges, 0.73);
        let rotated = describe_corners(&rotated_corners, &rotated_edges);
        for (a, b) in base[0].values.iter().zip(&rotated[0].values) {
            assert!((a - b).abs() < 1e-4, "descriptor drift {a} vs {b}");
        }
    }

    fn segment(x0: f32, y0: f32, x1: f32, y1: f32) -> Segment {
        let length = (x1 - x0).hypot(y1 - y0);
        let mut orientation = (y1 - y0).atan2(x1 - x0);
        if orientation < 0.0 {
            orientation += std::f32::consts::PI;
        }
        Segment {
            x0,
            y0,
            x1,
            y1,
            length,
            orientation,
            weight: length,
        }
    }

    #[test]
    fn segment_pair_invariants_are_similarity_invariant() {
        let segments = vec![
            segment(0.0, 0.0, 10.0, 0.0),
            segment(5.0, 4.0, 5.0, 12.0),
            segment(20.0, 0.0, 26.0, 6.0),
        ];
        let base = segment_pair_invariants(&segments, 500);

        let angle = 0.9f32;
        let (sin, cos) = angle.sin_cos();
        let transform = |x: f32, y: f32| [cos * x - sin * y + 3.0, sin * x + cos * y - 2.0];
        let transformed: Vec<Segment> = segments
            .iter()
            .map(|s| {
                let a = transform(s.x0, s.y0);
                let b = transform(s.x1, s.y1);
                segment(a[0], a[1], b[0], b[1])
            })
            .collect();
        let moved = segment_pair_invariants(&transformed, 500);
        assert_eq!(base.len(), moved.len());
        for (a, b) in base.iter().zip(&moved) {
            assert_eq!((a.first, a.second), (b.first, b.second));
            assert!((a.length_ratio - b.length_ratio).abs() < 1e-5);
            assert!((a.angle_between - b.angle_between).abs() < 1e-5);
            assert!((a.midpoint_distance - b.midpoint_distance).abs() < 1e-5);
            assert!((a.midpoint_angle - b.midpoint_angle).abs() < 1e-5);
        }
    }

    #[test]
    fn parallel_pairs_are_rejected() {
        let segments = vec![
            segment(0.0, 0.0, 10.0, 0.0),
            segment(0.0, 3.0, 8.0, 3.0),
            segment(0.0, 0.0, 0.0, 10.0),
        ];
        let pairs = segment_pair_invariants(&segments, 500);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|p| p.angle_between > 0.087));
    }
}
