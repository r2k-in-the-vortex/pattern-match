//! Exclusion-mask primitives and rasterization in model coordinates
//! (README §7.6, §8.2).
//!
//! Primitives are stored exactly as authored (polygons and brush strokes tagged
//! add/erase).  The raster is derived deterministically; it is not the
//! authoritative representation.  Features whose centers fall in an excluded
//! cell are removed from the model.

use serde::{Deserialize, Serialize};

use crate::geometry::ZeroFrame;

/// Maximum raster side used to bound memory for malformed bounds.
const MAX_MASK_SIDE: usize = 4096;

/// Axis-aligned bounds of a mask raster in model coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskBounds {
    /// Minimum x.
    pub min_x: f32,
    /// Minimum y.
    pub min_y: f32,
    /// Maximum x.
    pub max_x: f32,
    /// Maximum y.
    pub max_y: f32,
}

impl MaskBounds {
    /// Bounds around a set of points, expanded by `margin`.
    #[must_use]
    pub fn from_points(points: impl IntoIterator<Item = [f32; 2]>, margin: f32) -> Self {
        let mut bounds = Self {
            min_x: f32::INFINITY,
            min_y: f32::INFINITY,
            max_x: f32::NEG_INFINITY,
            max_y: f32::NEG_INFINITY,
        };
        let mut any = false;
        for point in points {
            any = true;
            bounds.min_x = bounds.min_x.min(point[0]);
            bounds.min_y = bounds.min_y.min(point[1]);
            bounds.max_x = bounds.max_x.max(point[0]);
            bounds.max_y = bounds.max_y.max(point[1]);
        }
        if !any {
            return Self {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 1.0,
                max_y: 1.0,
            };
        }
        bounds.min_x -= margin;
        bounds.min_y -= margin;
        bounds.max_x += margin;
        bounds.max_y += margin;
        if bounds.max_x <= bounds.min_x {
            bounds.max_x = bounds.min_x + 1.0;
        }
        if bounds.max_y <= bounds.min_y {
            bounds.max_y = bounds.min_y + 1.0;
        }
        bounds
    }

    /// Combine two bounds.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Whether the bounds are finite and non-empty.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min_x.is_finite()
            && self.min_y.is_finite()
            && self.max_x.is_finite()
            && self.max_y.is_finite()
            && self.max_x > self.min_x
            && self.max_y > self.min_y
    }
}

/// A mask-edit primitive in model coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaskPrimitive {
    /// A closed polygon; `add` excludes its interior, erase clears it.
    Polygon {
        /// Polygon vertices.
        points: Vec<[f32; 2]>,
        /// `true` adds exclusion, `false` erases it.
        add: bool,
    },
    /// A brush stroke: a polyline swept with `radius`.
    Brush {
        /// Stroke centerline points.
        points: Vec<[f32; 2]>,
        /// Brush radius in model pixels.
        radius: f32,
        /// `true` adds exclusion, `false` erases it.
        add: bool,
    },
}

impl MaskPrimitive {
    /// Whether this primitive adds or erases exclusion.
    #[must_use]
    pub fn is_add(&self) -> bool {
        match self {
            Self::Polygon { add, .. } | Self::Brush { add, .. } => *add,
        }
    }

    /// The primitive's points.
    #[must_use]
    pub fn points(&self) -> &[[f32; 2]] {
        match self {
            Self::Polygon { points, .. } | Self::Brush { points, .. } => points,
        }
    }

    /// Whether the primitive can contribute to a raster.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self {
            Self::Polygon { points, .. } => {
                points.len() >= 3 && points.iter().all(|p| p[0].is_finite() && p[1].is_finite())
            }
            Self::Brush { points, radius, .. } => {
                !points.is_empty()
                    && radius.is_finite()
                    && *radius > 0.0
                    && points.iter().all(|p| p[0].is_finite() && p[1].is_finite())
            }
        }
    }

    /// Convert the primitive from model to image coordinates.
    #[must_use]
    pub fn to_image(&self, zero: &ZeroFrame) -> Self {
        match self {
            Self::Polygon { points, add } => Self::Polygon {
                points: points.iter().map(|&p| zero.to_image(p)).collect(),
                add: *add,
            },
            Self::Brush {
                points,
                radius,
                add,
            } => Self::Brush {
                points: points.iter().map(|&p| zero.to_image(p)).collect(),
                radius: *radius,
                add: *add,
            },
        }
    }

    /// Convert the primitive from image to model coordinates.
    #[must_use]
    pub fn to_model(&self, zero: &ZeroFrame) -> Self {
        match self {
            Self::Polygon { points, add } => Self::Polygon {
                points: points.iter().map(|&p| zero.to_model(p)).collect(),
                add: *add,
            },
            Self::Brush {
                points,
                radius,
                add,
            } => Self::Brush {
                points: points.iter().map(|&p| zero.to_model(p)).collect(),
                radius: *radius,
                add: *add,
            },
        }
    }

    /// Scale the primitive's coordinates and brush radius by `factor`.
    #[must_use]
    pub fn scaled(&self, factor: f32) -> Self {
        match self {
            Self::Polygon { points, add } => Self::Polygon {
                points: points
                    .iter()
                    .map(|point| [point[0] * factor, point[1] * factor])
                    .collect(),
                add: *add,
            },
            Self::Brush {
                points,
                radius,
                add,
            } => Self::Brush {
                points: points
                    .iter()
                    .map(|point| [point[0] * factor, point[1] * factor])
                    .collect(),
                radius: radius * factor,
                add: *add,
            },
        }
    }

    /// Bounds of the primitive in its current coordinates, or `None` when
    /// invalid.
    #[must_use]
    pub fn bounds(&self) -> Option<MaskBounds> {
        if !self.is_valid() {
            return None;
        }
        let margin = match self {
            Self::Polygon { .. } => 0.0,
            Self::Brush { radius, .. } => *radius,
        };
        Some(MaskBounds::from_points(
            self.points().iter().copied(),
            margin,
        ))
    }
}

/// Rasterized binary exclusion mask over [`MaskBounds`].
#[derive(Clone, Debug, PartialEq)]
pub struct MaskRaster {
    /// Model-coordinate bounds of the raster.
    pub bounds: MaskBounds,
    /// Pixel size in model units.
    pub resolution: f32,
    /// Raster width in cells.
    pub width: usize,
    /// Raster height in cells.
    pub height: usize,
    /// Row-major exclusion flags.
    pub excluded: Vec<bool>,
}

impl MaskRaster {
    /// An empty raster over the given bounds.
    pub fn empty(bounds: MaskBounds, resolution: f32) -> Self {
        Self::rasterize(&[], bounds, resolution)
    }

    /// Rasterize primitives in order.  Later primitives override earlier ones.
    #[must_use]
    pub fn rasterize(primitives: &[MaskPrimitive], bounds: MaskBounds, resolution: f32) -> Self {
        let mut resolution = if resolution.is_finite() && resolution > 0.0 {
            resolution
        } else {
            1.0
        };
        // Guard against pathological bounds from malformed files by coarsening
        // the raster rather than truncating its coverage.
        let width_units = (bounds.max_x - bounds.min_x).max(0.0);
        let height_units = (bounds.max_y - bounds.min_y).max(0.0);
        let longest = (width_units / resolution).max(height_units / resolution);
        if longest > MAX_MASK_SIDE as f32 {
            resolution *= longest / MAX_MASK_SIDE as f32;
        }
        let width = ((width_units / resolution).ceil() as usize).clamp(1, MAX_MASK_SIDE);
        let height = ((height_units / resolution).ceil() as usize).clamp(1, MAX_MASK_SIDE);
        let mut excluded = vec![false; width * height];
        let raster = Self {
            bounds,
            resolution,
            width,
            height,
            excluded: Vec::new(),
        };
        for primitive in primitives {
            if !primitive.is_valid() {
                continue;
            }
            match primitive {
                MaskPrimitive::Polygon { points, add } => {
                    let cells = raster.cells_for_points(points, 0.0);
                    for (x, y) in cells {
                        if polygon_contains(points, raster.cell_center(x, y)) {
                            excluded[y * width + x] = *add;
                        }
                    }
                }
                MaskPrimitive::Brush {
                    points,
                    radius,
                    add,
                } => {
                    let cells = raster.cells_for_points(points, *radius);
                    for (x, y) in cells {
                        let center = raster.cell_center(x, y);
                        if distance_to_polyline(center, points) <= *radius {
                            excluded[y * width + x] = *add;
                        }
                    }
                }
            }
        }
        Self {
            bounds,
            resolution,
            width,
            height,
            excluded,
        }
    }

    /// Center of a raster cell in model coordinates.
    #[must_use]
    pub fn cell_center(&self, x: usize, y: usize) -> [f32; 2] {
        [
            self.bounds.min_x + (x as f32 + 0.5) * self.resolution,
            self.bounds.min_y + (y as f32 + 0.5) * self.resolution,
        ]
    }

    /// Whether a model point is excluded.  Points outside the raster are kept.
    #[must_use]
    pub fn is_excluded(&self, p: [f32; 2]) -> bool {
        if !p[0].is_finite() || !p[1].is_finite() {
            return false;
        }
        let x = ((p[0] - self.bounds.min_x) / self.resolution).floor();
        let y = ((p[1] - self.bounds.min_y) / self.resolution).floor();
        if x < 0.0 || y < 0.0 {
            return false;
        }
        let (x, y) = (x as usize, y as usize);
        if x >= self.width || y >= self.height {
            return false;
        }
        self.excluded[y * self.width + x]
    }

    /// Number of excluded cells.
    #[must_use]
    pub fn excluded_count(&self) -> usize {
        self.excluded.iter().filter(|&&v| v).count()
    }

    /// Stable hash of the raster dimensions, resolution, and exclusion flags.
    #[must_use]
    pub fn hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |byte: u8| {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for byte in (self.width as u32).to_le_bytes() {
            mix(byte);
        }
        for byte in (self.height as u32).to_le_bytes() {
            mix(byte);
        }
        for byte in self.resolution.to_le_bytes() {
            mix(byte);
        }
        for &excluded in &self.excluded {
            mix(u8::from(excluded));
        }
        hash
    }

    /// Render the mask as an RGBA image for display (red where excluded).
    #[must_use]
    pub fn to_rgba(&self) -> image::RgbaImage {
        let mut image = image::RgbaImage::from_pixel(
            self.width as u32,
            self.height as u32,
            image::Rgba([0, 0, 0, 0]),
        );
        for (index, &excluded) in self.excluded.iter().enumerate() {
            if excluded {
                let x = (index % self.width) as u32;
                let y = (index / self.width) as u32;
                image.put_pixel(x, y, image::Rgba([220, 40, 40, 160]));
            }
        }
        image
    }

    fn cells_for_points(&self, points: &[[f32; 2]], margin: f32) -> Vec<(usize, usize)> {
        let min_x = points.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min) - margin;
        let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min) - margin;
        let max_x = points
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max)
            + margin;
        let max_y = points
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max)
            + margin;

        let x0 = (((min_x - self.bounds.min_x) / self.resolution).floor() as isize)
            .clamp(0, self.width as isize - 1) as usize;
        let y0 = (((min_y - self.bounds.min_y) / self.resolution).floor() as isize)
            .clamp(0, self.height as isize - 1) as usize;
        let x1 = (((max_x - self.bounds.min_x) / self.resolution).ceil() as isize)
            .clamp(0, self.width as isize - 1) as usize;
        let y1 = (((max_y - self.bounds.min_y) / self.resolution).ceil() as isize)
            .clamp(0, self.height as isize - 1) as usize;

        let mut cells = Vec::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                cells.push((x, y));
            }
        }
        cells
    }
}

fn polygon_contains(points: &[[f32; 2]], p: [f32; 2]) -> bool {
    let mut inside = false;
    let mut j = points.len() - 1;
    for i in 0..points.len() {
        let (xi, yi) = (points[i][0], points[i][1]);
        let (xj, yj) = (points[j][0], points[j][1]);
        if (yi > p[1]) != (yj > p[1]) && p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn distance_to_polyline(p: [f32; 2], points: &[[f32; 2]]) -> f32 {
    if points.len() == 1 {
        return (p[0] - points[0][0]).hypot(p[1] - points[0][1]);
    }
    let mut best = f32::INFINITY;
    for pair in points.windows(2) {
        best = best.min(distance_to_segment(p, pair[0], pair[1]));
    }
    best
}

fn distance_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let length_sq = abx * abx + aby * aby;
    if length_sq <= 1e-12 {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    let t = (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length_sq).clamp(0.0, 1.0);
    let q = [a[0] + t * abx, a[1] + t * aby];
    (p[0] - q[0]).hypot(p[1] - q[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds() -> MaskBounds {
        MaskBounds::from_points([[0.0, 0.0], [10.0, 10.0]], 0.0)
    }

    #[test]
    fn polygon_rasterization_adds_and_erases() {
        let add = MaskPrimitive::Polygon {
            points: vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]],
            add: true,
        };
        let erase = MaskPrimitive::Polygon {
            points: vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]],
            add: false,
        };
        let raster = MaskRaster::rasterize(&[add, erase], bounds(), 1.0);
        assert!(raster.is_excluded([3.0, 3.0]));
        assert!(!raster.is_excluded([5.0, 5.0]));
        assert!(!raster.is_excluded([0.5, 0.5]));
        assert!(raster.excluded_count() > 0);
    }

    #[test]
    fn brush_stroke_excludes_within_radius() {
        let brush = MaskPrimitive::Brush {
            points: vec![[2.0, 5.0], [8.0, 5.0]],
            radius: 1.5,
            add: true,
        };
        let raster = MaskRaster::rasterize(&[brush], bounds(), 1.0);
        assert!(raster.is_excluded([5.0, 5.0]));
        assert!(raster.is_excluded([5.0, 6.0]));
        assert!(!raster.is_excluded([5.0, 8.0]));
    }

    #[test]
    fn erase_after_add_restores_cells() {
        let add = MaskPrimitive::Brush {
            points: vec![[5.0, 5.0]],
            radius: 3.0,
            add: true,
        };
        let erase = MaskPrimitive::Brush {
            points: vec![[5.0, 5.0]],
            radius: 1.0,
            add: false,
        };
        let raster = MaskRaster::rasterize(&[add, erase], bounds(), 1.0);
        assert!(!raster.is_excluded([5.0, 5.0]));
        assert!(raster.is_excluded([5.0, 7.0]));
    }

    #[test]
    fn primitive_coordinate_conversion_roundtrip() {
        let zero = ZeroFrame {
            origin: [12.0, -3.0],
            angle: 0.4,
        };
        let primitive = MaskPrimitive::Brush {
            points: vec![[2.0, 3.0], [-1.0, 4.0]],
            radius: 2.0,
            add: true,
        };
        let model = primitive.to_model(&zero);
        let image = model.to_image(&zero);
        match (&primitive, &image) {
            (
                MaskPrimitive::Brush {
                    points: a,
                    radius: ra,
                    ..
                },
                MaskPrimitive::Brush {
                    points: b,
                    radius: rb,
                    ..
                },
            ) => {
                assert!((ra - rb).abs() < 1e-6);
                for (pa, pb) in a.iter().zip(b) {
                    assert!((pa[0] - pb[0]).abs() < 1e-5);
                    assert!((pa[1] - pb[1]).abs() < 1e-5);
                }
            }
            _ => panic!("kind changed"),
        }
    }

    #[test]
    fn mask_serde_roundtrip() {
        let primitives = vec![
            MaskPrimitive::Polygon {
                points: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                add: true,
            },
            MaskPrimitive::Brush {
                points: vec![[2.0, 2.0]],
                radius: 1.0,
                add: false,
            },
        ];
        let json = serde_json::to_string(&primitives).unwrap();
        let decoded: Vec<MaskPrimitive> = serde_json::from_str(&json).unwrap();
        assert_eq!(primitives, decoded);
    }
}
