//! Feature records, extraction parameters, and the full-resolution extraction
//! pipeline (README §7).  Mask rasterization and exclusion arrive with the
//! teaching tools in M3.

use serde::{Deserialize, Serialize};

use crate::corner::{CornerParams, detect_corners};
use crate::edge::{EdgeParams, extract_edges};
use crate::gradient::GradientField;
use crate::gray::GrayF32;
use crate::segment::{SegmentParams, fit_segments};

/// A subpixel edge point with a canonical normal and signed polarity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgePoint {
    /// Subpixel x coordinate.
    pub x: f32,
    /// Subpixel y coordinate.
    pub y: f32,
    /// Canonical normal x (upper half-plane, README §7.1).
    pub nx: f32,
    /// Canonical normal y.
    pub ny: f32,
    /// Gradient magnitude at the edge.
    pub magnitude: f32,
    /// Polarity: `+1` when intensity increases along the normal, `-1` otherwise.
    pub sign: i8,
    /// Chain identifier; points in the same connected chain share it.
    pub chain_id: u32,
    /// Weight used by later scoring stages; currently the gradient magnitude.
    pub weight: f32,
}

/// An ordered, 8-connected edge chain.  Entries index into [`EdgeSet::points`].
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeChain {
    /// Chain identifier.
    pub id: u32,
    /// Point indices in chain order.
    pub points: Vec<usize>,
}

/// Edge points and their chains.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EdgeSet {
    /// Edge points.
    pub points: Vec<EdgePoint>,
    /// Chains containing at least the minimum number of points.
    pub chains: Vec<EdgeChain>,
}

/// A Shi–Tomasi corner with subpixel refinement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// Subpixel x coordinate.
    pub x: f32,
    /// Subpixel y coordinate.
    pub y: f32,
    /// Shi–Tomasi response (`λ_min`) at the integer peak.
    pub response: f32,
    /// Dominant orientation of the structure tensor in radians, modulo `π`.
    pub orientation: f32,
    /// Weight used by later stages.
    pub weight: f32,
}

/// A line segment fitted to an edge chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// First subpixel endpoint.
    pub x0: f32,
    /// First subpixel endpoint.
    pub y0: f32,
    /// Second subpixel endpoint.
    pub x1: f32,
    /// Second subpixel endpoint.
    pub y1: f32,
    /// Euclidean endpoint distance in pixels.
    pub length: f32,
    /// Undirected orientation in radians, in `[0, π)`.
    pub orientation: f32,
    /// Weight used by later stages; currently the length.
    pub weight: f32,
}

/// The feature set extracted from one image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureSet {
    /// Edge points and chains.
    pub edges: EdgeSet,
    /// Corners.
    pub corners: Vec<Corner>,
    /// Line segments.
    pub segments: Vec<Segment>,
}

impl FeatureSet {
    /// Number of edge points.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.points.len()
    }

    /// Number of edge chains.
    #[must_use]
    pub fn chain_count(&self) -> usize {
        self.edges.chains.len()
    }

    /// Number of corners.
    #[must_use]
    pub fn corner_count(&self) -> usize {
        self.corners.len()
    }

    /// Number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// Parameters for the full feature extraction pipeline.
///
/// Defaults follow README Appendix A.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureParams {
    /// Full-resolution Gaussian blur before gradients.
    pub blur_sigma: f32,
    /// Edge extraction parameters.
    pub edge: EdgeParams,
    /// Corner extraction parameters.
    pub corner: CornerParams,
    /// Segment fitting parameters.
    pub segment: SegmentParams,
    /// Whether to run corner detection.
    pub detect_corners: bool,
    /// Whether to fit line segments.
    pub fit_segments: bool,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            blur_sigma: 1.0,
            edge: EdgeParams::default(),
            corner: CornerParams::default(),
            segment: SegmentParams::default(),
            detect_corners: true,
            fit_segments: true,
        }
    }
}

/// Extract edges, corners, and segments from a full-resolution grayscale image.
///
/// The pipeline is deterministic: identical inputs and parameters produce
/// identical outputs.
#[must_use]
pub fn extract_features(image: &GrayF32, params: &FeatureParams) -> FeatureSet {
    let blurred = if params.blur_sigma > 0.01 {
        image.gaussian_blur(params.blur_sigma)
    } else {
        image.clone()
    };
    let gradient = GradientField::compute(&blurred);
    let edges = extract_edges(&gradient, &params.edge);
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
    FeatureSet {
        edges,
        corners,
        segments,
    }
}

/// Keep only features whose representative point satisfies `keep`.
///
/// Edge points are filtered individually; chains are remapped and empty chains
/// are removed.  Corners use their center; segments use their midpoint.
#[must_use]
pub fn filter_features(
    features: &FeatureSet,
    mut keep: impl FnMut(f32, f32) -> bool,
) -> FeatureSet {
    let mut point_keep = vec![false; features.edges.points.len()];
    for (index, point) in features.edges.points.iter().enumerate() {
        point_keep[index] = keep(point.x, point.y);
    }

    let mut kept_chains = Vec::new();
    let mut referenced = vec![false; features.edges.points.len()];
    for chain in &features.edges.chains {
        let mut indices = Vec::with_capacity(chain.points.len());
        for &index in &chain.points {
            if point_keep.get(index).copied().unwrap_or(false) {
                referenced[index] = true;
                indices.push(index);
            }
        }
        if !indices.is_empty() {
            kept_chains.push(EdgeChain {
                id: chain.id,
                points: indices,
            });
        }
    }

    let mut old_to_new = vec![usize::MAX; features.edges.points.len()];
    let mut points = Vec::new();
    for (index, point) in features.edges.points.iter().enumerate() {
        if referenced[index] {
            old_to_new[index] = points.len();
            points.push(*point);
        }
    }
    for chain in &mut kept_chains {
        chain.points = chain
            .points
            .iter()
            .map(|&index| old_to_new[index])
            .collect();
    }

    let corners = features
        .corners
        .iter()
        .copied()
        .filter(|corner| keep(corner.x, corner.y))
        .collect();
    let segments = features
        .segments
        .iter()
        .copied()
        .filter(|segment| {
            keep(
                (segment.x0 + segment.x1) * 0.5,
                (segment.y0 + segment.y1) * 0.5,
            )
        })
        .collect();

    FeatureSet {
        edges: EdgeSet {
            points,
            chains: kept_chains,
        },
        corners,
        segments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plate_image(width: usize, height: usize) -> GrayF32 {
        // A bright rectangle on a dark background with anti-aliased edges.
        let mut data = vec![0.1f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32;
                let fy = y as f32;
                if (10.0..width as f32 - 10.0).contains(&fx)
                    && (10.0..height as f32 - 10.0).contains(&fy)
                {
                    data[y * width + x] = 0.9;
                }
            }
        }
        GrayF32::new(width, height, data).unwrap()
    }

    #[test]
    fn pipeline_finds_edges_corners_and_segments() {
        let image = plate_image(80, 64);
        let features = extract_features(&image, &FeatureParams::default());
        assert!(
            features.edge_count() > 100,
            "edges: {}",
            features.edge_count()
        );
        assert!(features.chain_count() > 0);
        assert!(
            features.corner_count() >= 4,
            "corners: {}",
            features.corner_count()
        );
        assert!(
            features.segment_count() >= 4,
            "segments: {}",
            features.segment_count()
        );
    }

    #[test]
    fn pipeline_is_deterministic() {
        let image = plate_image(48, 40);
        let params = FeatureParams::default();
        let first = extract_features(&image, &params);
        let second = extract_features(&image, &params);
        assert_eq!(first, second);
    }

    #[test]
    fn feature_toggles_are_respected() {
        let image = plate_image(48, 40);
        let params = FeatureParams {
            detect_corners: false,
            fit_segments: false,
            ..FeatureParams::default()
        };
        let features = extract_features(&image, &params);
        assert!(features.edge_count() > 0);
        assert_eq!(features.corner_count(), 0);
        assert_eq!(features.segment_count(), 0);
    }

    #[test]
    fn filter_features_remaps_chains() {
        let image = plate_image(80, 64);
        let features = extract_features(&image, &FeatureParams::default());
        let filtered = filter_features(&features, |x, _| x < 40.0);
        assert!(filtered.edge_count() > 0);
        assert!(filtered.edges.points.iter().all(|p| p.x < 40.0));
        assert!(filtered.corners.iter().all(|c| c.x < 40.0));
        for chain in &filtered.edges.chains {
            assert!(!chain.points.is_empty());
            for &index in &chain.points {
                assert!(index < filtered.edges.points.len());
                assert_eq!(
                    filtered.edges.points[index].chain_id, chain.id,
                    "chain id mismatch"
                );
            }
        }
    }
}
