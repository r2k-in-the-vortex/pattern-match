//! Feature overlay rendering (README §3.3, §7.5).
//!
//! The overlay is a transparent RGBA image so a UI or exporter can composite it
//! over the source image.  Edge hue encodes the canonical normal orientation;
//! corners use a fixed accent color; segments are cyan.

use image::{Rgba, RgbaImage};

use crate::feature::FeatureSet;

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
}
