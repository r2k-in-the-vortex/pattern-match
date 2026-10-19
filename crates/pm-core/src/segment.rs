//! Line-segment fitting over ordered edge chains using split-and-merge
//! (README §7.3).

use crate::feature::{EdgeSet, Segment};

/// Segment fitting parameters.  Defaults follow README Appendix A.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SegmentParams {
    /// Minimum endpoint distance in pixels.
    pub min_length: f32,
    /// Maximum perpendicular deviation of a point from its fitted line.
    pub max_perpendicular_error: f32,
    /// Minimum number of points in an input chain.
    pub min_points: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            min_length: 10.0,
            max_perpendicular_error: 0.75,
            min_points: 4,
        }
    }
}

/// Fit line segments to all eligible edge chains.
#[must_use]
pub fn fit_segments(edges: &EdgeSet, params: &SegmentParams) -> Vec<Segment> {
    let mut segments = Vec::new();
    for chain in &edges.chains {
        let points: Vec<(f32, f32)> = chain
            .points
            .iter()
            .filter_map(|&index| edges.points.get(index))
            .map(|point| (point.x, point.y))
            .collect();
        if points.len() < params.min_points.max(4) {
            continue;
        }

        let mut ranges = Vec::new();
        split_range(&points, 0, points.len(), params, &mut ranges);
        merge_ranges(&points, params, &mut ranges);

        for (start, end) in ranges {
            if end.saturating_sub(start) < 2 {
                continue;
            }
            if let Some(segment) = make_segment(&points[start..end], params.min_length) {
                segments.push(segment);
            }
        }
    }
    segments
}

/// Fit a line by total least squares (PCA).
struct LineFit {
    mean: (f32, f32),
    direction: (f32, f32),
    max_error: f32,
}

fn fit_line(points: &[(f32, f32)]) -> LineFit {
    let count = points.len() as f32;
    let mean = (
        points.iter().map(|p| p.0).sum::<f32>() / count,
        points.iter().map(|p| p.1).sum::<f32>() / count,
    );
    let (mut cxx, mut cxy, mut cyy) = (0.0f32, 0.0f32, 0.0f32);
    for &(x, y) in points {
        let dx = x - mean.0;
        let dy = y - mean.1;
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
    }
    let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let direction = (angle.cos(), angle.sin());
    let normal = (-direction.1, direction.0);

    let mut max_error = 0.0f32;
    for &(x, y) in points {
        let error = ((x - mean.0) * normal.0 + (y - mean.1) * normal.1).abs();
        max_error = max_error.max(error);
    }
    LineFit {
        mean,
        direction,
        max_error,
    }
}

fn split_range(
    points: &[(f32, f32)],
    start: usize,
    end: usize,
    params: &SegmentParams,
    out: &mut Vec<(usize, usize)>,
) {
    let count = end - start;
    let (max_error, split_index) = chord_max_error(points, start, end);
    if max_error > params.max_perpendicular_error
        && count >= 4
        && split_index > 0
        && split_index + 1 < count
    {
        let middle = start + split_index;
        split_range(points, start, middle + 1, params, out);
        split_range(points, middle, end, params, out);
    } else {
        out.push((start, end));
    }
}

/// Maximum perpendicular distance from the chord joining the first and last
/// point, used for the split decision.
fn chord_max_error(points: &[(f32, f32)], start: usize, end: usize) -> (f32, usize) {
    let (x0, y0) = points[start];
    let (x1, y1) = points[end - 1];
    let dx = x1 - x0;
    let dy = y1 - y0;
    let length = dx.hypot(dy);
    let mut max_error = 0.0f32;
    let mut split_index = 0;
    if length <= 1e-6 {
        for (index, &(x, y)) in points[start..end].iter().enumerate() {
            let error = (x - x0).hypot(y - y0);
            if error > max_error {
                max_error = error;
                split_index = index;
            }
        }
    } else {
        let nx = -dy / length;
        let ny = dx / length;
        for (index, &(x, y)) in points[start..end].iter().enumerate() {
            let error = ((x - x0) * nx + (y - y0) * ny).abs();
            if error > max_error {
                max_error = error;
                split_index = index;
            }
        }
    }
    (max_error, split_index)
}

/// Merge adjacent ranges that remain collinear within tolerance.
fn merge_ranges(points: &[(f32, f32)], params: &SegmentParams, ranges: &mut Vec<(usize, usize)>) {
    let mut index = 0;
    while index + 1 < ranges.len() {
        let (start, _) = ranges[index];
        let (_, end) = ranges[index + 1];
        if end - start >= 2
            && fit_line(&points[start..end]).max_error <= params.max_perpendicular_error
        {
            ranges[index] = (start, end);
            ranges.remove(index + 1);
        } else {
            index += 1;
        }
    }
}

fn make_segment(points: &[(f32, f32)], min_length: f32) -> Option<Segment> {
    if points.len() < 2 {
        return None;
    }
    let fit = fit_line(points);
    let (mut min_t, mut max_t) = (f32::INFINITY, f32::NEG_INFINITY);
    for &(x, y) in points {
        let t = (x - fit.mean.0) * fit.direction.0 + (y - fit.mean.1) * fit.direction.1;
        min_t = min_t.min(t);
        max_t = max_t.max(t);
    }
    let x0 = fit.mean.0 + fit.direction.0 * min_t;
    let y0 = fit.mean.1 + fit.direction.1 * min_t;
    let x1 = fit.mean.0 + fit.direction.0 * max_t;
    let y1 = fit.mean.1 + fit.direction.1 * max_t;
    let length = (x1 - x0).hypot(y1 - y0);
    if length < min_length {
        return None;
    }
    let mut orientation = (y1 - y0).atan2(x1 - x0);
    if orientation < 0.0 {
        orientation += std::f32::consts::PI;
    }
    if orientation >= std::f32::consts::PI {
        orientation -= std::f32::consts::PI;
    }
    Some(Segment {
        x0,
        y0,
        x1,
        y1,
        length,
        orientation,
        weight: length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{EdgeChain, EdgePoint, EdgeSet};

    fn edges_from_chain(points: &[(f32, f32)]) -> EdgeSet {
        let edge_points: Vec<EdgePoint> = points
            .iter()
            .map(|&(x, y)| EdgePoint {
                x,
                y,
                nx: 0.0,
                ny: -1.0,
                magnitude: 1.0,
                sign: 1,
                chain_id: 0,
                weight: 1.0,
            })
            .collect();
        let indices = (0..edge_points.len()).collect();
        EdgeSet {
            points: edge_points,
            chains: vec![EdgeChain {
                id: 0,
                points: indices,
            }],
        }
    }

    #[test]
    fn fits_straight_chain() {
        let points: Vec<(f32, f32)> = (0..40)
            .map(|i| {
                let t = i as f32 * 0.5;
                (t, 2.0 * t + 3.0)
            })
            .collect();
        let edges = edges_from_chain(&points);
        let segments = fit_segments(&edges, &SegmentParams::default());
        assert_eq!(segments.len(), 1, "segments: {segments:?}");
        let segment = segments[0];
        let expected_length = 19.5 * 5.0f32.sqrt();
        assert!(
            (segment.length - expected_length).abs() < 0.5,
            "length {}",
            segment.length
        );
        let expected = 2.0f32.atan2(1.0);
        assert!((segment.orientation - expected).abs() < 0.02);
        let start_error = (segment.x0).hypot(segment.y0 - 3.0);
        assert!(start_error < 0.5, "start error {start_error}");
    }

    #[test]
    fn splits_curved_chain() {
        let points: Vec<(f32, f32)> = (0..60)
            .map(|i| {
                let x = i as f32 * 0.5;
                (x, 0.05 * x * x)
            })
            .collect();
        let edges = edges_from_chain(&points);
        let segments = fit_segments(&edges, &SegmentParams::default());
        assert!(segments.len() >= 2, "expected split, got {segments:?}");
        for segment in &segments {
            assert!(segment.length >= 10.0);
            assert!(segment.length.is_finite());
        }
    }

    #[test]
    fn rejects_short_chain() {
        let points = vec![(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)];
        let edges = edges_from_chain(&points);
        assert!(fit_segments(&edges, &SegmentParams::default()).is_empty());
    }

    #[test]
    fn filters_short_segments() {
        let points: Vec<(f32, f32)> = (0..8).map(|i| (i as f32 * 0.5, 0.0)).collect();
        let edges = edges_from_chain(&points);
        let params = SegmentParams {
            min_length: 20.0,
            ..SegmentParams::default()
        };
        assert!(fit_segments(&edges, &params).is_empty());
    }
}
