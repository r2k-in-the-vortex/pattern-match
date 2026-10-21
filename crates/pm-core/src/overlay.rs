//! Feature overlay rendering (README §3.3, §7.5).
//!
//! The overlay is a transparent RGBA image so a UI or exporter can composite it
//! over the source image.  Edge hue encodes the canonical normal orientation;
//! corners use a fixed accent color; segments are cyan.

use image::{Rgba, RgbaImage};
use nalgebra::Vector2;

use crate::feature::{EdgePoint, FeatureSet};
use crate::geometry::Homography;
use crate::score::Detection;

/// Which feature layers to draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayOptions {
    /// Draw edge points as thin tangent strokes.
    pub draw_edges: bool,
    /// Draw corners as small crosses.
    pub draw_corners: bool,
    /// Draw line segments.
    pub draw_segments: bool,
    /// Total length of each edge tangent stroke in pixels.
    pub edge_stroke_length: f32,
    /// Half-size of the corner cross in pixels.
    pub corner_radius: i32,
}

impl Default for OverlayOptions {
    fn default() -> Self {
        Self {
            draw_edges: true,
            draw_corners: true,
            draw_segments: true,
            edge_stroke_length: 3.0,
            corner_radius: 2,
        }
    }
}

/// Fixed accent color for corners.
pub const CORNER_COLOR: Rgba<u8> = Rgba([255, 80, 60, 255]);
/// Segment color.
pub const SEGMENT_COLOR: Rgba<u8> = Rgba([0, 220, 220, 255]);
/// Matched model edge points in a detection overlay.
pub const MATCHED_EDGE_COLOR: Rgba<u8> = Rgba([70, 255, 160, 255]);
/// Detection zero-position marker.
pub const ZERO_MARKER_COLOR: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// Rotating outline colors for detections.
pub const DETECTION_COLORS: [Rgba<u8>; 6] = [
    Rgba([80, 255, 120, 255]),
    Rgba([80, 180, 255, 255]),
    Rgba([255, 200, 60, 255]),
    Rgba([255, 110, 220, 255]),
    Rgba([120, 255, 255, 255]),
    Rgba([255, 140, 80, 255]),
];

/// Which layers [`render_detection_overlay`] draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionOverlayOptions {
    /// Draw the transformed ROI outline.
    pub draw_roi: bool,
    /// Draw the model zero origin and axes.
    pub draw_axes: bool,
    /// Draw matched model edge points.
    pub draw_matched_edges: bool,
    /// Keep every `n`-th model edge point; zero is treated as one.
    pub matched_edge_stride: usize,
    /// Length of each zero-frame axis in model units; non-positive derives it
    /// from the ROI size.
    pub axis_length: f32,
}

impl Default for DetectionOverlayOptions {
    fn default() -> Self {
        Self {
            draw_roi: true,
            draw_axes: true,
            draw_matched_edges: true,
            matched_edge_stride: 1,
            axis_length: 0.0,
        }
    }
}

/// Composite detection geometry over a display image (README §3.3, §12.2).
///
/// `roi_model` and `model_edges` are in model coordinates; each detection's
/// reported homography maps them into the image.  The result is an opaque copy
/// of `base` with the requested layers drawn on top.
#[must_use]
pub fn render_detection_overlay(
    base: &RgbaImage,
    roi_model: &[[f32; 2]; 4],
    model_edges: &[EdgePoint],
    detections: &[Detection],
    options: &DetectionOverlayOptions,
) -> RgbaImage {
    let mut image = base.clone();
    let default_axis = roi_axis_length(roi_model);
    let axis_length = if options.axis_length > 0.0 {
        options.axis_length
    } else {
        default_axis
    };
    let stride = options.matched_edge_stride.max(1);

    for (index, detection) in detections.iter().enumerate() {
        let homography = Homography::from_array(detection.homography);
        let color = DETECTION_COLORS[index % DETECTION_COLORS.len()];
        let transform =
            |p: [f32; 2]| homography.try_act_point(Vector2::new(f64::from(p[0]), f64::from(p[1])));

        if options.draw_matched_edges {
            for point in model_edges.iter().step_by(stride) {
                if let Some(q) = transform([point.x, point.y]) {
                    put_pixel(
                        &mut image,
                        q.x.round() as i32,
                        q.y.round() as i32,
                        MATCHED_EDGE_COLOR,
                    );
                }
            }
        }

        if options.draw_roi {
            let corners: Vec<Vector2<f64>> = roi_model
                .iter()
                .filter_map(|corner| transform(*corner))
                .collect();
            if corners.len() == roi_model.len() {
                for pair in 0..corners.len() {
                    let start = corners[pair];
                    let end = corners[(pair + 1) % corners.len()];
                    draw_line(
                        &mut image,
                        start.x as f32,
                        start.y as f32,
                        end.x as f32,
                        end.y as f32,
                        color,
                    );
                }
            }
        }

        if options.draw_axes
            && let Some(origin) = transform([0.0, 0.0])
        {
            if let Some(x_axis) = transform([axis_length, 0.0]) {
                draw_line(
                    &mut image,
                    origin.x as f32,
                    origin.y as f32,
                    x_axis.x as f32,
                    x_axis.y as f32,
                    color,
                );
            }
            if let Some(y_axis) = transform([0.0, axis_length]) {
                draw_line(
                    &mut image,
                    origin.x as f32,
                    origin.y as f32,
                    y_axis.x as f32,
                    y_axis.y as f32,
                    color,
                );
            }
            let x = origin.x.round() as i32;
            let y = origin.y.round() as i32;
            for offset in -2..=2 {
                put_pixel(&mut image, x + offset, y, ZERO_MARKER_COLOR);
                put_pixel(&mut image, x, y + offset, ZERO_MARKER_COLOR);
            }
        }
    }

    image
}

fn roi_axis_length(roi_model: &[[f32; 2]; 4]) -> f32 {
    let width = (roi_model[1][0] - roi_model[0][0]).hypot(roi_model[1][1] - roi_model[0][1]);
    let height = (roi_model[3][0] - roi_model[0][0]).hypot(roi_model[3][1] - roi_model[0][1]);
    width.min(height) * 0.5
}

/// Color for an edge normal orientation: hue maps the undirected normal angle
/// (`[0, π)`) onto the HSV wheel.
#[must_use]
pub fn orientation_color(angle_rad: f32) -> [u8; 3] {
    let hue = angle_rad.rem_euclid(std::f32::consts::PI) / std::f32::consts::PI;
    hsv_to_rgb(hue, 0.85, 0.95)
}

/// Render a transparent overlay containing the requested feature layers.
#[must_use]
pub fn render_overlay(
    width: u32,
    height: u32,
    features: &FeatureSet,
    options: &OverlayOptions,
) -> RgbaImage {
    let mut image = RgbaImage::from_pixel(width, height, Rgba([0, 0, 0, 0]));
    if width == 0 || height == 0 {
        return image;
    }

    if options.draw_edges {
        let half = options.edge_stroke_length.max(1.0) * 0.5;
        for point in &features.edges.points {
            let color = orientation_color(point.ny.atan2(point.nx));
            // Tangent is the normal rotated by 90°.
            let tx = -point.ny;
            let ty = point.nx;
            draw_line(
                &mut image,
                point.x - tx * half,
                point.y - ty * half,
                point.x + tx * half,
                point.y + ty * half,
                Rgba([color[0], color[1], color[2], 255]),
            );
        }
    }

    if options.draw_segments {
        for segment in &features.segments {
            draw_line(
                &mut image,
                segment.x0,
                segment.y0,
                segment.x1,
                segment.y1,
                SEGMENT_COLOR,
            );
        }
    }

    if options.draw_corners {
        for corner in &features.corners {
            let x = corner.x.round() as i32;
            let y = corner.y.round() as i32;
            for offset in -options.corner_radius..=options.corner_radius {
                put_pixel(&mut image, x + offset, y, CORNER_COLOR);
                put_pixel(&mut image, x, y + offset, CORNER_COLOR);
            }
        }
    }

    image
}

fn draw_line(image: &mut RgbaImage, x0: f32, y0: f32, x1: f32, y1: f32, color: Rgba<u8>) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as usize;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let x = x0 + dx * t;
        let y = y0 + dy * t;
        put_pixel(image, x.round() as i32, y.round() as i32, color);
    }
}

fn put_pixel(image: &mut RgbaImage, x: i32, y: i32, color: Rgba<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < image.width() && (y as u32) < image.height() {
        image.put_pixel(x as u32, y as u32, color);
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let sector = h.floor() as i32;
    let f = h - sector as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let (r, g, b) = match sector % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{Corner, EdgeChain, EdgePoint, EdgeSet, Segment};

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

    fn features() -> FeatureSet {
        FeatureSet {
            edges: EdgeSet {
                points: vec![edge_point(10.0, 10.0, 1.0, 0.0)],
                chains: vec![EdgeChain {
                    id: 0,
                    points: vec![0],
                }],
            },
            corners: vec![Corner {
                x: 20.0,
                y: 20.0,
                response: 1.0,
                orientation: 0.0,
                weight: 1.0,
            }],
            segments: vec![Segment {
                x0: 5.0,
                y0: 5.0,
                x1: 5.0,
                y1: 15.0,
                length: 10.0,
                orientation: std::f32::consts::FRAC_PI_2,
                weight: 10.0,
            }],
        }
    }

    #[test]
    fn empty_overlay_is_transparent() {
        let overlay = render_overlay(16, 16, &FeatureSet::default(), &OverlayOptions::default());
        assert_eq!(overlay.dimensions(), (16, 16));
        assert!(overlay.pixels().all(|p| p[3] == 0));
    }

    #[test]
    fn edge_hue_follows_normal_orientation() {
        let red = orientation_color(0.0);
        assert!(red[0] > red[1] && red[0] > red[2], "hue 0 is red: {red:?}");
        let cyan = orientation_color(std::f32::consts::FRAC_PI_2);
        assert!(
            cyan[1] > cyan[0] && cyan[2] > cyan[0],
            "hue 0.5 is cyan: {cyan:?}"
        );
        let wrapped = orientation_color(std::f32::consts::PI - 1e-6);
        assert!(wrapped[0] > wrapped[1] && wrapped[0] > wrapped[2]);
    }

    #[test]
    fn draws_thin_tangent_strokes() {
        let overlay = render_overlay(32, 32, &features(), &OverlayOptions::default());
        // Normal (1, 0) => vertical tangent through (10, 10).
        for y in 9..=11 {
            assert!(overlay.get_pixel(10, y)[3] > 0, "pixel (10,{y})");
        }
        // Thin: adjacent columns are untouched by that stroke.
        assert_eq!(overlay.get_pixel(9, 10)[3], 0);
        assert_eq!(overlay.get_pixel(11, 10)[3], 0);
    }

    #[test]
    fn draws_corner_cross_and_segment() {
        let overlay = render_overlay(32, 32, &features(), &OverlayOptions::default());
        assert_eq!(overlay.get_pixel(20, 20), &CORNER_COLOR);
        assert_eq!(overlay.get_pixel(22, 20), &CORNER_COLOR);
        assert_eq!(overlay.get_pixel(5, 10), &SEGMENT_COLOR);
    }

    #[test]
    fn layer_toggles_disable_layers() {
        let options = OverlayOptions {
            draw_edges: false,
            draw_corners: false,
            draw_segments: false,
            ..OverlayOptions::default()
        };
        let overlay = render_overlay(32, 32, &features(), &options);
        assert!(overlay.pixels().all(|p| p[3] == 0));
    }

    #[test]
    fn rendering_does_not_mutate_features() {
        let features = features();
        let clone = features.clone();
        let _ = render_overlay(32, 32, &features, &OverlayOptions::default());
        assert_eq!(features, clone);
    }

    fn detection(homography: [[f64; 3]; 3]) -> Detection {
        Detection {
            id: 0,
            pose: crate::geometry::Similarity::from_pose(16.0, 16.0, 0.0, 1.0),
            homography,
            homography_estimated: true,
            score: 0.9,
            confidence: 0.9,
            diagnostics: crate::score::DetectionDiagnostics::default(),
        }
    }

    #[test]
    fn detection_overlay_draws_roi_axes_and_edges() {
        let base = RgbaImage::from_pixel(32, 32, Rgba([10, 10, 10, 255]));
        let roi_model = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
        let edges = vec![
            edge_point(3.0, 3.0, 1.0, 0.0),
            edge_point(4.0, 3.0, 1.0, 0.0),
        ];
        let homography = [[1.0, 0.0, 16.0], [0.0, 1.0, 16.0], [0.0, 0.0, 1.0]];
        let overlay = render_detection_overlay(
            &base,
            &roi_model,
            &edges,
            &[detection(homography)],
            &DetectionOverlayOptions::default(),
        );
        assert_eq!(overlay.dimensions(), (32, 32));
        // ROI corner (-5, -5) maps to (11, 11).
        assert_eq!(overlay.get_pixel(11, 11), &DETECTION_COLORS[0]);
        // Zero marker arms reach two pixels from the origin.
        assert_eq!(overlay.get_pixel(14, 16), &ZERO_MARKER_COLOR);
        // Matched model edges map to (19, 19) and (20, 19).
        assert_eq!(overlay.get_pixel(19, 19), &MATCHED_EDGE_COLOR);
        assert_eq!(overlay.get_pixel(20, 19), &MATCHED_EDGE_COLOR);
        // Untouched background is preserved.
        assert_eq!(overlay.get_pixel(0, 0), &Rgba([10, 10, 10, 255]));
    }

    #[test]
    fn detection_overlay_layer_toggles() {
        let base = RgbaImage::from_pixel(32, 32, Rgba([10, 10, 10, 255]));
        let roi_model = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
        let options = DetectionOverlayOptions {
            draw_roi: false,
            draw_axes: false,
            draw_matched_edges: false,
            ..DetectionOverlayOptions::default()
        };
        let overlay = render_detection_overlay(
            &base,
            &roi_model,
            &[edge_point(0.0, 0.0, 1.0, 0.0)],
            &[detection([
                [1.0, 0.0, 16.0],
                [0.0, 1.0, 16.0],
                [0.0, 0.0, 1.0],
            ])],
            &options,
        );
        assert_eq!(overlay, base);
    }
}
