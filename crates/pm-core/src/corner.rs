//! Shi–Tomasi corner response, NMS, and subpixel refinement (README §7.2).

use serde::{Deserialize, Serialize};

use crate::feature::Corner;
use crate::gradient::GradientField;
use crate::gray::GrayF32;

/// Corner extraction parameters.  Defaults follow README Appendix A.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CornerParams {
    /// Accepted response as a fraction of the strongest response.
    pub quality: f32,
    /// Minimum accepted distance between corners in pixels.
    pub min_distance: usize,
    /// Standard deviation of the structure-tensor integration window.
    pub window_sigma: f32,
}

impl Default for CornerParams {
    fn default() -> Self {
        Self {
            quality: 0.01,
            min_distance: 5,
            window_sigma: 1.5,
        }
    }
}

/// Detect Shi–Tomasi corners with subpixel refinement.
#[must_use]
pub fn detect_corners(gradient: &GradientField, params: &CornerParams) -> Vec<Corner> {
    let (width, height) = (gradient.width, gradient.height);
    if width == 0 || height == 0 {
        return Vec::new();
    }

    // Structure tensor with a Gaussian integration window.
    let mut sxx = Vec::with_capacity(width * height);
    let mut syy = Vec::with_capacity(width * height);
    let mut sxy = Vec::with_capacity(width * height);
    for i in 0..width * height {
        let (gx, gy) = (gradient.gx[i], gradient.gy[i]);
        sxx.push(gx * gx);
        syy.push(gy * gy);
        sxy.push(gx * gy);
    }
    let sxx = GrayF32::new(width, height, sxx)
        .expect("valid dimensions")
        .gaussian_blur(params.window_sigma);
    let syy = GrayF32::new(width, height, syy)
        .expect("valid dimensions")
        .gaussian_blur(params.window_sigma);
    let sxy = GrayF32::new(width, height, sxy)
        .expect("valid dimensions")
        .gaussian_blur(params.window_sigma);

    let response: Vec<f32> = (0..width * height)
        .map(|i| {
            let a = sxx.data[i];
            let b = sxy.data[i];
            let c = syy.data[i];
            let lambda_min = 0.5 * (a + c) - (0.25 * (a - c) * (a - c) + b * b).sqrt();
            lambda_min.max(0.0)
        })
        .collect();
    let max_response = response.iter().copied().fold(0.0f32, f32::max);
    if max_response <= 0.0 {
        return Vec::new();
    }

    let threshold = params.quality.max(0.0) * max_response;
    let mut candidates: Vec<usize> = (0..width * height)
        .filter(|&i| response[i] > threshold)
        .collect();
    candidates.sort_by(|&a, &b| response[b].total_cmp(&response[a]).then_with(|| a.cmp(&b)));

    let min_distance_sq = (params.min_distance * params.min_distance) as f32;
    let mut accepted: Vec<(usize, usize)> = Vec::new();
    for index in candidates {
        let x = index % width;
        let y = index / width;
        let too_close = accepted.iter().any(|&(ax, ay)| {
            let dx = (ax as f32 - x as f32).powi(2);
            let dy = (ay as f32 - y as f32).powi(2);
            dx + dy < min_distance_sq
        });
        if !too_close {
            accepted.push((x, y));
        }
    }

    accepted
        .into_iter()
        .map(|(x, y)| {
            let (sub_x, sub_y) = subpixel_offset(&response, width, height, x, y);
            let index = y * width + x;
            let orientation =
                0.5 * (2.0 * sxy.data[index]).atan2(sxx.data[index] - syy.data[index]);
            Corner {
                x: x as f32 + sub_x,
                y: y as f32 + sub_y,
                response: response[index],
                orientation,
                weight: response[index],
            }
        })
        .collect()
}

/// Quadratic-fit subpixel offset of the response peak.  Accepted only for a
/// strictly concave fit with a bounded offset.
fn subpixel_offset(
    response: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> (f32, f32) {
    if x == 0 || y == 0 || x + 1 >= width || y + 1 >= height {
        return (0.0, 0.0);
    }
    let r = |dx: isize, dy: isize| {
        response[(y as isize + dy) as usize * width + (x as isize + dx) as usize]
    };
    let center = r(0, 0);
    let xm = r(-1, 0);
    let xp = r(1, 0);
    let ym = r(0, -1);
    let yp = r(0, 1);
    let xmym = r(-1, -1);
    let xpym = r(1, -1);
    let xmyp = r(-1, 1);
    let xpyp = r(1, 1);

    let dxx = xp - 2.0 * center + xm;
    let dyy = yp - 2.0 * center + ym;
    let dxy = (xpyp - xmyp - xpym + xmym) / 4.0;
    let gx = (xp - xm) / 2.0;
    let gy = (yp - ym) / 2.0;
    let determinant = dxx * dyy - dxy * dxy;
    if dxx < 0.0 && determinant > 1e-12 {
        let offset_x = -(dyy * gx - dxy * gy) / determinant;
        let offset_y = -(-dxy * gx + dxx * gy) / determinant;
        if offset_x.is_finite()
            && offset_y.is_finite()
            && offset_x.abs() <= 0.5
            && offset_y.abs() <= 0.5
        {
            return (offset_x, offset_y);
        }
    }

    // Fallback: independent one-dimensional parabola fits along each axis.
    // This is more robust near the image border or where the cross term makes
    // the full Hessian slightly indefinite.
    let offset_x = if dxx < -1e-12 {
        (xm - xp) / (2.0 * dxx)
    } else {
        0.0
    };
    let offset_y = if dyy < -1e-12 {
        (ym - yp) / (2.0 * dyy)
    } else {
        0.0
    };
    if offset_x.is_finite()
        && offset_y.is_finite()
        && offset_x.abs() <= 0.5
        && offset_y.abs() <= 0.5
    {
        (offset_x, offset_y)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(
        width: usize,
        height: usize,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        w: f32,
    ) -> GrayF32 {
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            let sy = ((y as f32 - y0) / w).tanh() * ((y as f32 - y1) / w).tanh();
            for x in 0..width {
                let sx = ((x as f32 - x0) / w).tanh() * ((x as f32 - x1) / w).tanh();
                data.push(0.5 + 0.35 * sx * sy);
            }
        }
        GrayF32::new(width, height, data).unwrap()
    }

    fn nearest_corner(corners: &[Corner], x: f32, y: f32) -> f32 {
        corners
            .iter()
            .map(|c| ((c.x - x).powi(2) + (c.y - y).powi(2)).sqrt())
            .fold(f32::INFINITY, f32::min)
    }

    #[test]
    fn detects_smooth_checkerboard_corners_subpixel() {
        let image = checkerboard(80, 64, 16.3, 60.7, 14.2, 48.6, 1.2);
        let gradient = GradientField::compute(&image);
        let corners = detect_corners(&gradient, &CornerParams::default());
        assert!(!corners.is_empty());
        let expected = [(16.3, 14.2), (60.7, 14.2), (16.3, 48.6), (60.7, 48.6)];
        for (x, y) in expected {
            let error = nearest_corner(&corners, x, y);
            assert!(
                error < 0.4,
                "corner ({x},{y}) error {error}; corners {corners:?}"
            );
        }
    }

    #[test]
    fn flat_image_has_no_corners() {
        let image = GrayF32::filled(32, 32, 0.5);
        let gradient = GradientField::compute(&image);
        assert!(detect_corners(&gradient, &CornerParams::default()).is_empty());
    }

    #[test]
    fn minimum_distance_suppresses_duplicates() {
        let image = checkerboard(64, 64, 20.0, 44.0, 20.0, 44.0, 1.0);
        let gradient = GradientField::compute(&image);
        let corners = detect_corners(
            &gradient,
            &CornerParams {
                min_distance: 12,
                ..CornerParams::default()
            },
        );
        for (i, a) in corners.iter().enumerate() {
            for b in &corners[i + 1..] {
                let distance = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
                assert!(distance >= 12.0 - 1e-3, "corners too close: {distance}");
            }
        }
    }
}
