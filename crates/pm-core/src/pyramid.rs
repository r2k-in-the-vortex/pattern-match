//! Anti-aliased Gaussian pyramid (README §6).
//!
//! Level 0 is the input image.  Each subsequent level is built by blurring the
//! previous level with `σ = 1.0` and taking every second pixel.  Coordinates
//! convert between levels with `q_L = q_0 / 2^L`.

use crate::gray::GrayF32;

/// Level count used by default (`L_max = 3`, image/8).
pub const DEFAULT_LEVELS: usize = 3;

/// The smallest accepted short side of a pyramid level.
pub const MIN_LEVEL_SIZE: usize = 16;

/// A Gaussian pyramid of grayscale images.
#[derive(Clone, Debug)]
pub struct Pyramid {
    levels: Vec<GrayF32>,
}

impl Pyramid {
    /// Build a pyramid with at most `max_levels` non-base levels (README §6).
    ///
    /// The default blur is `σ = 1.0`.  Building stops early when the next level
    /// would be smaller than [`MIN_LEVEL_SIZE`] on either side.
    #[must_use]
    pub fn build(image: &GrayF32, max_levels: usize) -> Self {
        Self::build_with_sigma(image, max_levels, 1.0)
    }

    /// Build a pyramid with an explicit per-level blur.
    #[must_use]
    pub fn build_with_sigma(image: &GrayF32, max_levels: usize, sigma: f32) -> Self {
        let mut levels = vec![image.clone()];
        for _ in 0..max_levels {
            let current = levels.last().expect("base level exists");
            if current.is_empty()
                || current.width / 2 < MIN_LEVEL_SIZE
                || current.height / 2 < MIN_LEVEL_SIZE
            {
                break;
            }
            levels.push(current.gaussian_blur(sigma).downsample2());
        }
        Self { levels }
    }

    /// Number of levels including level 0.
    #[must_use]
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Whether the pyramid has no levels (only possible for an empty input).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// A level, or `None` when out of range.
    #[must_use]
    pub fn level(&self, level: usize) -> Option<&GrayF32> {
        self.levels.get(level)
    }

    /// A level, panicking when out of range.
    #[must_use]
    pub fn level_unchecked(&self, level: usize) -> &GrayF32 {
        &self.levels[level]
    }

    /// Full-resolution coordinate to level-`level` coordinate.
    #[must_use]
    pub fn to_level(x: f32, y: f32, level: usize) -> (f32, f32) {
        let factor = (1u32 << level) as f32;
        (x / factor, y / factor)
    }

    /// Level-`level` coordinate to full-resolution coordinate.
    #[must_use]
    pub fn from_level(x: f32, y: f32, level: usize) -> (f32, f32) {
        let factor = (1u32 << level) as f32;
        (x * factor, y * factor)
    }

    /// Downscale factor of a level.
    #[must_use]
    pub fn downscale(level: usize) -> f32 {
        (1u32 << level) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_default_levels_and_sizes() {
        let image = GrayF32::filled(128, 128, 0.5);
        let pyramid = Pyramid::build(&image, DEFAULT_LEVELS);
        assert_eq!(pyramid.len(), DEFAULT_LEVELS + 1);
        assert_eq!(
            (
                pyramid.level(1).unwrap().width,
                pyramid.level(1).unwrap().height
            ),
            (64, 64)
        );
        assert_eq!(
            (
                pyramid.level(3).unwrap().width,
                pyramid.level(3).unwrap().height
            ),
            (16, 16)
        );
    }

    #[test]
    fn stops_below_minimum_size() {
        let image = GrayF32::filled(20, 20, 0.5);
        let pyramid = Pyramid::build(&image, 5);
        assert_eq!(pyramid.len(), 1);
    }

    #[test]
    fn constant_image_stays_constant() {
        let image = GrayF32::filled(40, 40, 0.25);
        let pyramid = Pyramid::build(&image, 2);
        for level in 0..pyramid.len() {
            for &value in &pyramid.level(level).unwrap().data {
                assert!((value - 0.25).abs() < 1e-5, "level {level}");
            }
        }
    }

    #[test]
    fn coordinates_round_trip() {
        let (x, y) = Pyramid::from_level(3.5, -2.0, 3);
        assert_eq!((x, y), (28.0, -16.0));
        let (lx, ly) = Pyramid::to_level(x, y, 3);
        assert_eq!((lx, ly), (3.5, -2.0));
    }

    #[test]
    fn impulse_stays_centered_after_decimation() {
        let mut image = GrayF32::filled(64, 64, 0.0);
        image.data[32 * 64 + 32] = 1.0;
        let pyramid = Pyramid::build(&image, 1);
        let level = pyramid.level(1).unwrap();
        let center = level.data[16 * level.width + 16];
        assert!(center > 0.05, "center response {center}");
        assert!(
            (level.data[16 * level.width + 15] - level.data[16 * level.width + 17]).abs() < 1e-6
        );
        assert!(
            (level.data[15 * level.width + 16] - level.data[17 * level.width + 16]).abs() < 1e-6
        );
    }
}
