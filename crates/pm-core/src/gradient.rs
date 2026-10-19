//! Scharr gradients, magnitude, orientation, and signed polarity (README §6).

use crate::gray::GrayF32;

/// Threshold below which a gradient is treated as zero for orientation.
pub const GRADIENT_EPS: f32 = 1e-6;

/// Tolerance for treating a normal component as aligned with an axis.
const CANONICAL_EPS: f32 = 1e-6;

/// Dense gradient field for one image level.
#[derive(Clone, Debug)]
pub struct GradientField {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Horizontal derivative, row-major.
    pub gx: Vec<f32>,
    /// Vertical derivative (positive downward), row-major.
    pub gy: Vec<f32>,
    /// Gradient magnitude, row-major.
    pub magnitude: Vec<f32>,
}

impl GradientField {
    /// Compute Scharr gradients with edge-clamped sampling.
    ///
    /// The kernels are normalized by `1/32`, so a unit ramp has a derivative of
    /// approximately 1 per pixel.
    #[must_use]
    pub fn compute(image: &GrayF32) -> Self {
        let (width, height) = (image.width, image.height);
        let mut gx = vec![0.0f32; width * height];
        let mut gy = vec![0.0f32; width * height];
        let mut magnitude = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let xi = x as isize;
                let yi = y as isize;
                let p = |dx: isize, dy: isize| image.at_clamped(xi + dx, yi + dy);

                let gxv = (-3.0 * p(-1, -1) + 3.0 * p(1, -1) - 10.0 * p(-1, 0) + 10.0 * p(1, 0)
                    - 3.0 * p(-1, 1)
                    + 3.0 * p(1, 1))
                    / 32.0;
                let gyv = (-3.0 * p(-1, -1) - 10.0 * p(0, -1) - 3.0 * p(1, -1)
                    + 3.0 * p(-1, 1)
                    + 10.0 * p(0, 1)
                    + 3.0 * p(1, 1))
                    / 32.0;

                let index = y * width + x;
                gx[index] = gxv;
                gy[index] = gyv;
                magnitude[index] = gxv.hypot(gyv);
            }
        }

        Self {
            width,
            height,
            gx,
            gy,
            magnitude,
        }
    }

    /// Gradient at a pixel, or `None` outside the image.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> Option<(f32, f32)> {
        if x < self.width && y < self.height {
            let index = y * self.width + x;
            Some((self.gx[index], self.gy[index]))
        } else {
            None
        }
    }

    /// Magnitude at a pixel.
    #[must_use]
    pub fn magnitude_at(&self, x: usize, y: usize) -> f32 {
        self.magnitude[y * self.width + x]
    }

    /// Raw gradient orientation in radians, `atan2(gy, gx)`.
    #[must_use]
    pub fn orientation_at(&self, x: usize, y: usize) -> f32 {
        let index = y * self.width + x;
        self.gy[index].atan2(self.gx[index])
    }

    /// Canonical edge normal and signed polarity at a pixel.
    ///
    /// The normal is chosen in the upper half-plane (`ny > 0`, or `ny == 0` and
    /// `nx > 0`), so its direction is unique up to the half-turn ambiguity of
    /// an undirected edge.  `sign` is `+1` when the gradient points along the
    /// canonical normal and `-1` when it points the other way.  Returns `None`
    /// for a (numerically) zero gradient.
    #[must_use]
    pub fn normal_at(&self, x: usize, y: usize) -> Option<(f32, f32, i8)> {
        let index = y * self.width + x;
        let (gx, gy) = (self.gx[index], self.gy[index]);
        let magnitude = gx.hypot(gy);
        if magnitude <= GRADIENT_EPS {
            return None;
        }
        let mut nx = gx / magnitude;
        let mut ny = gy / magnitude;
        let mut sign = 1i8;
        if ny < -CANONICAL_EPS || (ny.abs() <= CANONICAL_EPS && nx < 0.0) {
            nx = -nx;
            ny = -ny;
            sign = -1;
        }
        Some((nx, ny, sign))
    }

    /// Bilinearly sample the magnitude at continuous coordinates.
    #[must_use]
    pub fn sample_magnitude(&self, x: f32, y: f32) -> f32 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let top = self.magnitude[y0 * self.width + x0]
            + (self.magnitude[y0 * self.width + x1] - self.magnitude[y0 * self.width + x0]) * fx;
        let bottom = self.magnitude[y1 * self.width + x0]
            + (self.magnitude[y1 * self.width + x1] - self.magnitude[y1 * self.width + x0]) * fx;
        top + (bottom - top) * fy
    }

    /// Bilinearly sample the gradient vector at continuous coordinates.
    #[must_use]
    pub fn sample(&self, x: f32, y: f32) -> (f32, f32) {
        if self.width == 0 || self.height == 0 {
            return (0.0, 0.0);
        }
        let x = x.clamp(0.0, (self.width - 1) as f32);
        let y = y.clamp(0.0, (self.height - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let sample = |plane: &[f32]| {
            let top = plane[y0 * self.width + x0]
                + (plane[y0 * self.width + x1] - plane[y0 * self.width + x0]) * fx;
            let bottom = plane[y1 * self.width + x0]
                + (plane[y1 * self.width + x1] - plane[y1 * self.width + x0]) * fx;
            top + (bottom - top) * fy
        };
        (sample(&self.gx), sample(&self.gy))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: usize, height: usize, step: f32) -> GrayF32 {
        let mut data = Vec::with_capacity(width * height);
        for _ in 0..height {
            for x in 0..width {
                data.push(step * x as f32);
            }
        }
        GrayF32::new(width, height, data).unwrap()
    }

    #[test]
    fn scharr_recovers_horizontal_ramp() {
        let image = ramp(32, 16, 0.01);
        let grad = GradientField::compute(&image);
        let (gx, gy) = grad.at(16, 8).unwrap();
        assert!((gx - 0.01).abs() < 1e-5, "gx {gx}");
        assert!(gy.abs() < 1e-6, "gy {gy}");
        let (nx, ny, sign) = grad.normal_at(16, 8).unwrap();
        assert!((nx - 1.0).abs() < 1e-6, "nx {nx}");
        assert!(ny.abs() < 1e-6, "ny {ny}");
        assert_eq!(sign, 1);
    }

    #[test]
    fn polarity_flips_with_contrast() {
        let rising = ramp(32, 16, 0.01);
        let falling = ramp(32, 16, -0.01);
        let rising = GradientField::compute(&rising);
        let falling = GradientField::compute(&falling);
        let (nx, _ny, sign) = rising.normal_at(16, 8).unwrap();
        assert!((nx - 1.0).abs() < 1e-6);
        assert_eq!(sign, 1);
        let (nx, _ny, sign) = falling.normal_at(16, 8).unwrap();
        assert!((nx - 1.0).abs() < 1e-6, "canonical normal is undirected");
        assert_eq!(sign, -1);
    }

    #[test]
    fn vertical_ramp_has_downward_gradient() {
        let mut data = Vec::new();
        for y in 0..16 {
            for _ in 0..32 {
                data.push(0.01 * y as f32);
            }
        }
        let image = GrayF32::new(32, 16, data).unwrap();
        let grad = GradientField::compute(&image);
        let (gx, gy) = grad.at(16, 8).unwrap();
        assert!(gx.abs() < 1e-6);
        assert!((gy - 0.01).abs() < 1e-5, "gy {gy}");
        let (nx, ny, sign) = grad.normal_at(16, 8).unwrap();
        assert!(nx.abs() < 1e-6);
        assert!((ny - 1.0).abs() < 1e-6);
        assert_eq!(sign, 1);
    }

    #[test]
    fn flat_image_has_no_normal() {
        let image = GrayF32::filled(8, 8, 0.3);
        let grad = GradientField::compute(&image);
        assert!(grad.normal_at(4, 4).is_none());
        assert!(grad.magnitude_at(4, 4) < 1e-7);
    }

    #[test]
    fn bilinear_gradient_sampling_matches_pixels() {
        let image = ramp(16, 16, 0.02);
        let grad = GradientField::compute(&image);
        let (gx, gy) = grad.sample(8.0, 8.0);
        assert!((gx - 0.02).abs() < 1e-5, "gx {gx}");
        assert!(gy.abs() < 1e-6);
        let mid = grad.sample_magnitude(8.5, 8.0);
        assert!((mid - 0.02).abs() < 1e-4, "mid {mid}");
    }
}
