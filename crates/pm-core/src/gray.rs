//! Grayscale image representation, conversions, sampling, ROI extraction, and
//! thumbnail generation (README §6).
//!
//! The internal working representation is [`GrayF32`] in `[0, 1]`. Color and
//! 16-bit inputs are converted on load; original pixel data is retained only in
//! [`LoadedImage::display`] for rendering. Grayscale conversion uses the BT.709
//! luma weights on encoded channel values (no linear-light conversion).

use std::path::Path;

use image::{DynamicImage, RgbaImage};
use rayon::prelude::*;

use crate::error::{Error, Result};

/// BT.709 luma weights applied to encoded RGB values.
const LUMA_R: f32 = 0.2126;
const LUMA_G: f32 = 0.7152;
const LUMA_B: f32 = 0.0722;

/// A single-channel `f32` image in `[0, 1]`, stored row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct GrayF32 {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Row-major pixel data, `width * height` values.
    pub data: Vec<f32>,
}

impl GrayF32 {
    /// Construct an image, validating the buffer length and dimensions.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidParameter(
                "image dimensions must be non-zero".to_owned(),
            ));
        }
        if data.len() != width * height {
            return Err(Error::InvalidParameter(format!(
                "pixel buffer has {} values, expected {}",
                data.len(),
                width * height
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Construct a constant image.
    ///
    /// # Panics
    ///
    /// Panics when a dimension is zero; use [`Self::new`] for untrusted input.
    #[must_use]
    pub fn filled(width: usize, height: usize, value: f32) -> Self {
        assert!(width > 0 && height > 0, "image dimensions must be non-zero");
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    /// Empty image with zero dimensions.  Useful only as a placeholder; all
    /// sampling and processing methods assume a non-empty image.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            width: 0,
            height: 0,
            data: Vec::new(),
        }
    }

    /// Whether the image has no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Pixel count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.width * self.height
    }

    /// Read a pixel, returning `None` outside the image.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        (x < self.width && y < self.height).then(|| self.data[y * self.width + x])
    }

    /// Read a pixel with edge clamping.  The image must be non-empty.
    #[must_use]
    pub fn at_clamped(&self, x: isize, y: isize) -> f32 {
        debug_assert!(!self.is_empty());
        let x = x.clamp(0, self.width as isize - 1) as usize;
        let y = y.clamp(0, self.height as isize - 1) as usize;
        self.data[y * self.width + x]
    }

    /// Bilinear sample at continuous image coordinates, clamping outside the
    /// image.  Integer coordinates are pixel centers.
    #[must_use]
    pub fn sample_bilinear(&self, x: f32, y: f32) -> f32 {
        if self.is_empty() {
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

        let v00 = self.data[y0 * self.width + x0];
        let v10 = self.data[y0 * self.width + x1];
        let v01 = self.data[y1 * self.width + x0];
        let v11 = self.data[y1 * self.width + x1];
        let top = v00 + (v10 - v00) * fx;
        let bottom = v01 + (v11 - v01) * fx;
        top + (bottom - top) * fy
    }

    /// Apply a separable Gaussian blur with standard deviation `sigma` pixels.
    /// Non-positive `sigma` returns a copy.
    #[must_use]
    pub fn gaussian_blur(&self, sigma: f32) -> Self {
        if self.is_empty() || !sigma.is_finite() || sigma <= 0.01 {
            return self.clone();
        }
        let kernel = gaussian_kernel(sigma);
        let radius = kernel.len() / 2;
        let (width, height) = (self.width, self.height);

        let mut horizontal = vec![0.0f32; self.len()];
        horizontal
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, value) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (k, weight) in kernel.iter().enumerate() {
                        let sx = x as isize + k as isize - radius as isize;
                        sum += weight * self.at_clamped(sx, y as isize);
                    }
                    *value = sum;
                }
            });

        let mut out = vec![0.0f32; self.len()];
        out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (k, weight) in kernel.iter().enumerate() {
                    let sy = y as isize + k as isize - radius as isize;
                    sum +=
                        weight * horizontal[sy.clamp(0, height as isize - 1) as usize * width + x];
                }
                *value = sum;
            }
        });
        Self {
            width: self.width,
            height: self.height,
            data: out,
        }
    }

    /// Halve the resolution by taking every second pixel after the caller has
    /// blurred the image.  Dimensions round up.
    #[must_use]
    pub fn downsample2(&self) -> Self {
        if self.is_empty() {
            return Self::empty();
        }
        let width = self.width.div_ceil(2);
        let height = self.height.div_ceil(2);
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                data.push(
                    self.data
                        [(2 * y).min(self.height - 1) * self.width + (2 * x).min(self.width - 1)],
                );
            }
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Crop an axis-aligned region.  Out-of-bounds pixels are filled with zero.
    pub fn crop(&self, x0: isize, y0: isize, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidParameter(
                "crop dimensions must be non-zero".to_owned(),
            ));
        }
        let mut data = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let sx = x0 + x as isize;
                let sy = y0 + y as isize;
                if sx >= 0 && sy >= 0 && (sx as usize) < self.width && (sy as usize) < self.height {
                    data[y * width + x] = self.data[sy as usize * self.width + sx as usize];
                }
            }
        }
        Self::new(width, height, data)
    }

    /// Bilinear resize to exactly `width × height`, mapping pixel centers.
    pub fn resize(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidParameter(
                "resize dimensions must be non-zero".to_owned(),
            ));
        }
        if self.is_empty() {
            return Err(Error::InvalidParameter(
                "cannot resize an empty image".to_owned(),
            ));
        }
        let sx = self.width as f32 / width as f32;
        let sy = self.height as f32 / height as f32;
        let mut data = Vec::with_capacity(width * height);
        for y in 0..height {
            let src_y = (y as f32 + 0.5) * sy - 0.5;
            for x in 0..width {
                let src_x = (x as f32 + 0.5) * sx - 0.5;
                data.push(self.sample_bilinear(src_x, src_y));
            }
        }
        Self::new(width, height, data)
    }

    /// Downscale to fit within `max_width × max_height`, preserving aspect
    /// ratio.  Images already small enough are returned unchanged.
    pub fn thumbnail(&self, max_width: usize, max_height: usize) -> Result<Self> {
        if self.is_empty() {
            return Err(Error::InvalidParameter(
                "cannot thumbnail an empty image".to_owned(),
            ));
        }
        let scale = (max_width as f32 / self.width as f32)
            .min(max_height as f32 / self.height as f32)
            .min(1.0);
        if scale >= 1.0 {
            return Ok(self.clone());
        }
        let width = ((self.width as f32 * scale).round() as usize).max(1);
        let height = ((self.height as f32 * scale).round() as usize).max(1);
        self.resize(width, height)
    }

    /// Render the image as opaque 8-bit RGBA for display.
    #[must_use]
    pub fn display_rgba(&self) -> RgbaImage {
        let mut out = Vec::with_capacity(self.len() * 4);
        for &value in &self.data {
            let v = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            out.extend_from_slice(&[v, v, v, 255]);
        }
        RgbaImage::from_raw(self.width as u32, self.height as u32, out)
            .expect("buffer length matches dimensions")
    }

    /// Convert a decoded image to grayscale using the README §6 rules.
    /// Alpha channels are ignored; 16-bit channels are scaled by `1/65535`.
    pub fn from_dynamic_image(image: &DynamicImage) -> Result<Self> {
        let (width, height) = (image.width() as usize, image.height() as usize);
        let data: Vec<f32> = match image {
            DynamicImage::ImageLuma8(img) => {
                img.as_raw().iter().map(|&v| v as f32 / 255.0).collect()
            }
            DynamicImage::ImageLumaA8(img) => img
                .as_raw()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| p[0] as f32 / 255.0)
                .collect(),
            DynamicImage::ImageRgb8(img) => img
                .as_raw()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| luma_u8(p[0], p[1], p[2]))
                .collect(),
            DynamicImage::ImageRgba8(img) => img
                .as_raw()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| luma_u8(p[0], p[1], p[2]))
                .collect(),
            DynamicImage::ImageLuma16(img) => {
                img.as_raw().iter().map(|&v| v as f32 / 65535.0).collect()
            }
            DynamicImage::ImageLumaA16(img) => img
                .as_raw()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| p[0] as f32 / 65535.0)
                .collect(),
            DynamicImage::ImageRgb16(img) => img
                .as_raw()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| luma_u16(p[0], p[1], p[2]))
                .collect(),
            DynamicImage::ImageRgba16(img) => img
                .as_raw()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| luma_u16(p[0], p[1], p[2]))
                .collect(),
            other => {
                return Err(Error::Image(format!(
                    "unsupported color type {:?}; expected 8-bit or 16-bit luma/RGB/RGBA",
                    other.color()
                )));
            }
        };
        Self::new(width, height, data)
    }

    /// Decode an image file (PNG/JPEG/BMP/TIFF, 8-bit or 16-bit).
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(load_image(path)?.gray)
    }
}

/// BT.709 luma from encoded 8-bit RGB.
#[must_use]
pub fn luma_u8(r: u8, g: u8, b: u8) -> f32 {
    (LUMA_R * r as f32 + LUMA_G * g as f32 + LUMA_B * b as f32) / 255.0
}

/// BT.709 luma from encoded 16-bit RGB.
#[must_use]
pub fn luma_u16(r: u16, g: u16, b: u16) -> f32 {
    (LUMA_R * r as f32 + LUMA_G * g as f32 + LUMA_B * b as f32) / 65535.0
}

/// A decoded image: grayscale working data plus an RGBA display copy.
#[derive(Clone, Debug)]
pub struct LoadedImage {
    /// Grayscale working representation.
    pub gray: GrayF32,
    /// Display copy retaining the original color channels (8-bit RGBA).
    pub display: RgbaImage,
}

impl LoadedImage {
    /// Build the working and display representations from a decoded image.
    pub fn from_dynamic_image(decoded: DynamicImage) -> Result<Self> {
        let gray = GrayF32::from_dynamic_image(&decoded)?;
        Ok(Self {
            gray,
            display: decoded.to_rgba8(),
        })
    }
}

/// Decode an image file into the working grayscale representation and a display
/// copy.  The display copy is 8-bit RGBA even for 16-bit inputs.
pub fn load_image(path: impl AsRef<Path>) -> Result<LoadedImage> {
    let path = path.as_ref();
    let decoded = image::ImageReader::open(path)
        .map_err(|err| Error::Image(format!("open {}: {err}", path.display())))?
        .with_guessed_format()
        .map_err(|err| Error::Image(format!("detect format {}: {err}", path.display())))?
        .decode()
        .map_err(|err| Error::Image(format!("decode {}: {err}", path.display())))?;
    LoadedImage::from_dynamic_image(decoded)
}

/// A normalized 1-D Gaussian kernel.
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let mut kernel = Vec::with_capacity(2 * radius + 1);
    let denom = 2.0 * sigma * sigma;
    for i in 0..=2 * radius {
        let x = i as f32 - radius as f32;
        kernel.push((-x * x / denom).exp());
    }
    let sum: f32 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    fn assert_close(actual: f32, expected: f32, context: &str) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "{context}: actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn reads_8bit_grayscale() {
        let img = GrayImage::from_raw(2, 1, vec![0, 255]).unwrap();
        let gray = GrayF32::from_dynamic_image(&DynamicImage::ImageLuma8(img)).unwrap();
        assert_close(gray.data[0], 0.0, "black");
        assert_close(gray.data[1], 1.0, "white");
    }

    #[test]
    fn reads_8bit_color_with_bt709_luma() {
        let img = RgbImage::from_raw(3, 1, vec![255, 0, 0, 0, 255, 0, 0, 0, 255]).unwrap();
        let gray = GrayF32::from_dynamic_image(&DynamicImage::ImageRgb8(img)).unwrap();
        assert_close(gray.data[0], LUMA_R, "red");
        assert_close(gray.data[1], LUMA_G, "green");
        assert_close(gray.data[2], LUMA_B, "blue");
    }

    #[test]
    fn reads_16bit_grayscale() {
        let img =
            image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(2, 1, vec![0, 65535]).unwrap();
        let gray = GrayF32::from_dynamic_image(&DynamicImage::ImageLuma16(img)).unwrap();
        assert_close(gray.data[0], 0.0, "16-bit black");
        assert_close(gray.data[1], 1.0, "16-bit white");
    }

    #[test]
    fn rejects_unsupported_float_images() {
        let img =
            image::ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(1, 1, vec![0.0, 0.0, 0.0]).unwrap();
        let err = GrayF32::from_dynamic_image(&DynamicImage::ImageRgb32F(img)).unwrap_err();
        assert!(matches!(err, Error::Image(_)));
    }

    #[test]
    fn bilinear_sampling_interpolates_and_clamps() {
        let gray = GrayF32::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]).unwrap();
        assert_close(gray.sample_bilinear(0.0, 0.0), 0.0, "corner");
        assert_close(gray.sample_bilinear(1.0, 0.0), 1.0, "corner");
        assert_close(gray.sample_bilinear(0.5, 0.5), 0.5, "center");
        assert_close(gray.sample_bilinear(-5.0, -5.0), 0.0, "clamped low");
        assert_close(gray.sample_bilinear(50.0, 50.0), 0.0, "clamped high");
    }

    #[test]
    fn blur_preserves_constant_and_smooths_impulse() {
        let constant = GrayF32::filled(9, 9, 0.4);
        let blurred = constant.gaussian_blur(1.0);
        for &v in &blurred.data {
            assert_close(v, 0.4, "constant preserved");
        }

        let mut impulse = GrayF32::filled(9, 9, 0.0);
        impulse.data[4 * 9 + 4] = 1.0;
        let blurred = impulse.gaussian_blur(1.0);
        assert!(blurred.data[4 * 9 + 4] > blurred.data[4 * 9 + 3]);
        assert!(blurred.data[4 * 9 + 3] > 0.0);
        assert!(
            (blurred.data[4 * 9 + 3] - blurred.data[4 * 9 + 5]).abs() < 1e-6,
            "horizontal symmetry"
        );
        assert!(
            (blurred.data[3 * 9 + 4] - blurred.data[5 * 9 + 4]).abs() < 1e-6,
            "vertical symmetry"
        );
    }

    #[test]
    fn downsample_halves_dimensions_and_rounds_up() {
        let gray = GrayF32::filled(5, 3, 0.25);
        let small = gray.downsample2();
        assert_eq!((small.width, small.height), (3, 2));
        assert!(small.data.iter().all(|&v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn thumbnail_preserves_aspect_ratio() {
        let gray = GrayF32::filled(100, 50, 0.5);
        let thumb = gray.thumbnail(40, 40).unwrap();
        assert_eq!((thumb.width, thumb.height), (40, 20));
        let unchanged = gray.thumbnail(100, 50).unwrap();
        assert_eq!((unchanged.width, unchanged.height), (100, 50));
    }

    #[test]
    fn crop_and_resize() {
        let gray = GrayF32::new(4, 3, (0..12).map(|i| i as f32).collect()).unwrap();
        let cropped = gray.crop(1, 1, 2, 2).unwrap();
        assert_eq!(cropped.data, vec![5.0, 6.0, 9.0, 10.0]);
        let resized = gray.resize(2, 2).unwrap();
        assert_eq!((resized.width, resized.height), (2, 2));
        assert_eq!(resized.data.len(), 4);
    }

    #[test]
    fn loads_png_from_disk() {
        let img = GrayImage::from_raw(2, 2, vec![0, 64, 128, 255]).unwrap();
        let path = std::env::temp_dir().join(format!("pm-gray-test-{}.png", std::process::id()));
        img.save(&path).unwrap();
        let loaded = load_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((loaded.gray.width, loaded.gray.height), (2, 2));
        assert_eq!(loaded.display.dimensions(), (2, 2));
        assert_close(loaded.gray.data[3], 1.0, "decoded white");
    }

    #[test]
    fn loads_16bit_png_from_disk() {
        let img =
            image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(2, 1, vec![0, 65535]).unwrap();
        let path =
            std::env::temp_dir().join(format!("pm-gray-16bit-test-{}.png", std::process::id()));
        DynamicImage::ImageLuma16(img).save(&path).unwrap();
        let loaded = load_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_close(loaded.gray.data[0], 0.0, "16-bit black");
        assert_close(loaded.gray.data[1], 1.0, "16-bit white");
    }
}
