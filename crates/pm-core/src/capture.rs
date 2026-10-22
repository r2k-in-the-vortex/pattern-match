//! Frame sources: the [`FrameSource`] trait, image files, and webcam capture
//! (README §5.4 and §11.4).
//!
//! A frame source is deliberately backend-neutral: it produces a
//! [`LoadedImage`] (grayscale working data plus an RGBA display copy) and a
//! sequence number. The UI owns threading; this module only opens devices and
//! pulls frames. Webcam support is compiled by the `camera` feature and uses
//! `nokhwa`'s Windows Media Foundation backend.
//!
//! Pixel formats are converted here rather than by `nokhwa` so the build does
//! not need its MJPEG decoder (and therefore no C/CMake dependency). MJPEG
//! frames are decoded with the pure-Rust `image` crate; YUYV, NV12, GRAY,
//! RAWRGB, and RAWBGR are converted directly.

use std::path::Path;
use std::time::Instant;

use crate::error::{Error, Result};
#[cfg(any(feature = "camera", test))]
use crate::gray::GrayF32;
use crate::gray::LoadedImage;

/// One frame delivered by a [`FrameSource`].
#[derive(Clone, Debug)]
pub struct Frame {
    /// Grayscale working image and display copy.
    pub image: LoadedImage,
    /// Monotonic source sequence, starting at 1.
    pub sequence: u64,
    /// Time the frame was received from the backend.
    pub captured_at: Instant,
}

/// A selectable capture device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureDevice {
    /// Zero-based device index accepted by the backend.
    pub index: u32,
    /// Human-readable device name.
    pub name: String,
    /// Backend description or symbolic link.
    pub description: String,
}

impl CaptureDevice {
    /// Label suitable for a device picker.
    #[must_use]
    pub fn label(&self) -> String {
        if self.name.trim().is_empty() {
            format!("Camera {}", self.index)
        } else {
            self.name.clone()
        }
    }
}

/// Requested webcam configuration.  The nearest supported mode is selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaptureConfig {
    /// Preferred frame width in pixels.
    pub width: u32,
    /// Preferred frame height in pixels.
    pub height: u32,
    /// Preferred frame rate.
    pub fps: u32,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
        }
    }
}

impl CaptureConfig {
    /// Whether the requested dimensions and rate are usable.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.width > 0 && self.height > 0 && self.fps > 0
    }
}

/// A source of [`Frame`]s.
///
/// `next_frame` blocks until a frame is available.  `Ok(None)` means the
/// source is exhausted (for example, a single still image) and the caller
/// should stop.  Implementations are owned by a capture thread; `stop` is
/// called after that thread leaves its loop.
pub trait FrameSource: Send {
    /// Human-readable description including the negotiated resolution/format.
    fn description(&self) -> String;

    /// Pull the next frame, blocking as needed.
    fn next_frame(&mut self) -> Result<Option<Frame>>;

    /// Release the underlying device.  Called once by the capture thread.
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A source that yields one still image and then ends.
pub struct ImageFileSource {
    image: Option<LoadedImage>,
    description: String,
    sequence: u64,
}

impl ImageFileSource {
    /// Load an image file as a one-frame source.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let image = crate::gray::load_image(path)?;
        let description = path.display().to_string();
        Ok(Self {
            image: Some(image),
            description,
            sequence: 0,
        })
    }

    /// Wrap an already decoded image as a one-frame source.
    #[must_use]
    pub fn from_image(image: LoadedImage, description: impl Into<String>) -> Self {
        Self {
            image: Some(image),
            description: description.into(),
            sequence: 0,
        }
    }
}

impl FrameSource for ImageFileSource {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn next_frame(&mut self) -> Result<Option<Frame>> {
        let Some(image) = self.image.take() else {
            return Ok(None);
        };
        self.sequence += 1;
        Ok(Some(Frame {
            image,
            sequence: self.sequence,
            captured_at: Instant::now(),
        }))
    }
}

/// Whether this build contains a usable camera backend.
#[must_use]
pub fn camera_available() -> bool {
    cfg!(all(feature = "camera", target_os = "windows"))
}

/// Enumerate available cameras.
///
/// The query runs on a fresh thread so the backend can initialize its COM
/// apartment without inheriting the UI thread's threading model.
#[allow(clippy::needless_return)]
pub fn list_cameras() -> Result<Vec<CaptureDevice>> {
    #[cfg(feature = "camera")]
    {
        return std::thread::spawn(|| {
            use nokhwa::utils::ApiBackend;
            let infos = nokhwa::query(ApiBackend::Auto)
                .map_err(|error| Error::Capture(format!("enumerate cameras: {error}")))?;
            Ok(infos
                .into_iter()
                .enumerate()
                .map(|(position, info)| {
                    let index = info
                        .index()
                        .as_index()
                        .unwrap_or_else(|_| u32::try_from(position).unwrap_or(0));
                    CaptureDevice {
                        index,
                        name: info.human_name(),
                        description: info.description().to_owned(),
                    }
                })
                .collect())
        })
        .join()
        .map_err(|_| Error::Capture("camera enumeration thread panicked".to_owned()))?;
    }
    #[cfg(not(feature = "camera"))]
    {
        Err(Error::Capture(
            "this build has no camera backend (enable the `camera` feature)".to_owned(),
        ))
    }
}

/// Open a webcam.  Called by the capture thread that will own it, so the
/// device is created and used in the same COM apartment.
#[allow(clippy::needless_return)]
pub fn open_webcam(device: &CaptureDevice, config: CaptureConfig) -> Result<Box<dyn FrameSource>> {
    if !config.is_valid() {
        return Err(Error::InvalidParameter(
            "camera width, height, and fps must be non-zero".to_owned(),
        ));
    }
    #[cfg(feature = "camera")]
    {
        return Ok(Box::new(WebcamSource::open(device, config)?));
    }
    #[cfg(not(feature = "camera"))]
    {
        let _ = device;
        Err(Error::Capture(
            "this build has no camera backend (enable the `camera` feature)".to_owned(),
        ))
    }
}

#[cfg(feature = "camera")]
mod webcam {
    use nokhwa::{
        Camera,
        utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType},
    };

    use super::*;

    /// Pixel formats this module can convert, in no particular preference order.
    const SUPPORTED_FORMATS: [FrameFormat; 6] = [
        FrameFormat::MJPEG,
        FrameFormat::YUYV,
        FrameFormat::NV12,
        FrameFormat::GRAY,
        FrameFormat::RAWRGB,
        FrameFormat::RAWBGR,
    ];

    /// A webcam frame source backed by `nokhwa`.
    pub(super) struct WebcamSource {
        camera: Camera,
        format: CameraFormat,
        sequence: u64,
    }

    impl WebcamSource {
        pub(super) fn open(device: &CaptureDevice, config: CaptureConfig) -> Result<Self> {
            let index = CameraIndex::Index(device.index);
            // The first negotiated format only lets us query the device; the
            // best supported mode is chosen from the full format list below.
            let mut camera = Camera::new(
                index,
                RequestedFormat::with_formats(RequestedFormatType::None, &SUPPORTED_FORMATS),
            )
            .map_err(|error| Error::Capture(format!("open {}: {error}", device.label())))?;

            let mut format = camera.camera_format();
            if let Ok(formats) = camera.compatible_camera_formats()
                && let Some(chosen) = choose_format(&formats, config)
            {
                let decoder = [chosen.format()];
                let request =
                    RequestedFormat::with_formats(RequestedFormatType::Exact(chosen), &decoder);
                match camera.set_camera_requset(request) {
                    Ok(applied) => format = applied,
                    Err(_) => format = camera.camera_format(),
                }
            }

            camera
                .open_stream()
                .map_err(|error| Error::Capture(format!("start stream: {error}")))?;
            Ok(Self {
                camera,
                format,
                sequence: 0,
            })
        }
    }

    impl FrameSource for WebcamSource {
        fn description(&self) -> String {
            format!(
                "{}×{} {} @ {} fps",
                self.format.width(),
                self.format.height(),
                self.format.format(),
                self.format.frame_rate()
            )
        }

        fn next_frame(&mut self) -> Result<Option<Frame>> {
            let buffer = self
                .camera
                .frame()
                .map_err(|error| Error::Capture(format!("read frame: {error}")))?;
            let captured_at = Instant::now();
            let image = buffer_to_loaded(&buffer)?;
            self.sequence += 1;
            Ok(Some(Frame {
                image,
                sequence: self.sequence,
                captured_at,
            }))
        }

        fn stop(&mut self) -> Result<()> {
            self.camera
                .stop_stream()
                .map_err(|error| Error::Capture(format!("stop stream: {error}")))
        }
    }

    /// Pick the supported mode closest to the requested size and rate.
    pub(super) fn choose_format(
        formats: &[CameraFormat],
        config: CaptureConfig,
    ) -> Option<CameraFormat> {
        formats
            .iter()
            .filter(|format| supported_format(format.format()))
            .copied()
            .min_by_key(|format| {
                let resolution = format.resolution();
                let width_delta =
                    (i64::from(resolution.width_x) - i64::from(config.width)).unsigned_abs();
                let height_delta =
                    (i64::from(resolution.height_y) - i64::from(config.height)).unsigned_abs();
                let fps_delta =
                    (i64::from(format.frame_rate()) - i64::from(config.fps)).unsigned_abs();
                (
                    width_delta + height_delta,
                    fps_delta,
                    format_rank(format.format()),
                    std::cmp::Reverse(
                        u64::from(resolution.width_x) * u64::from(resolution.height_y),
                    ),
                )
            })
    }

    /// All formats this module can convert.
    fn supported_format(format: FrameFormat) -> bool {
        SUPPORTED_FORMATS.contains(&format)
    }

    /// Tie-break preference: MJPEG is widely available and uses less USB
    /// bandwidth; the remaining formats follow in decode-cost order.
    fn format_rank(format: FrameFormat) -> u8 {
        match format {
            FrameFormat::MJPEG => 0,
            FrameFormat::YUYV => 1,
            FrameFormat::NV12 => 2,
            FrameFormat::GRAY => 3,
            FrameFormat::RAWRGB => 4,
            FrameFormat::RAWBGR => 5,
        }
    }

    /// Convert a nokhwa buffer into the working/display representation.
    fn buffer_to_loaded(buffer: &nokhwa::Buffer) -> Result<LoadedImage> {
        let resolution = buffer.resolution();
        let width = resolution.width_x as usize;
        let height = resolution.height_y as usize;
        if width == 0 || height == 0 {
            return Err(Error::Capture("camera returned an empty frame".to_owned()));
        }
        let data = buffer.buffer();
        match buffer.source_frame_format() {
            FrameFormat::MJPEG => {
                let decoded = image::load_from_memory(data)
                    .map_err(|error| Error::Capture(format!("decode MJPEG frame: {error}")))?;
                LoadedImage::from_dynamic_image(decoded)
            }
            FrameFormat::YUYV => {
                let rgba = yuyv_to_rgba(width, height, data)?;
                LoadedImage::from_dynamic_image(image::DynamicImage::ImageRgba8(rgba))
            }
            FrameFormat::NV12 => {
                let rgba = nv12_to_rgba(width, height, data)?;
                LoadedImage::from_dynamic_image(image::DynamicImage::ImageRgba8(rgba))
            }
            FrameFormat::GRAY => gray_to_loaded(width, height, data),
            FrameFormat::RAWRGB | FrameFormat::RAWBGR => {
                let rgba = raw_color_to_rgba(width, height, data, buffer.source_frame_format())?;
                LoadedImage::from_dynamic_image(image::DynamicImage::ImageRgba8(rgba))
            }
        }
    }

    pub(super) fn gray_to_loaded(width: usize, height: usize, data: &[u8]) -> Result<LoadedImage> {
        let stride = row_stride(data.len(), height, width)
            .ok_or_else(|| Error::Capture("GRAY frame has an inconsistent size".to_owned()))?;
        let mut values = Vec::with_capacity(width * height);
        for row in 0..height {
            let start = row * stride;
            values.extend(
                data[start..start + width]
                    .iter()
                    .map(|&value| f32::from(value) / 255.0),
            );
        }
        let gray = GrayF32::new(width, height, values)?;
        Ok(LoadedImage {
            display: gray.display_rgba(),
            gray,
        })
    }

    fn raw_color_to_rgba(
        width: usize,
        height: usize,
        data: &[u8],
        format: FrameFormat,
    ) -> Result<image::RgbaImage> {
        let stride = row_stride(data.len(), height, width * 3)
            .ok_or_else(|| Error::Capture("RGB frame has an inconsistent size".to_owned()))?;
        let mut rgba = vec![0_u8; width * height * 4];
        for row in 0..height {
            let source = &data[row * stride..row * stride + width * 3];
            let target = &mut rgba[row * width * 4..(row + 1) * width * 4];
            for column in 0..width {
                let input = &source[column * 3..column * 3 + 3];
                let output = &mut target[column * 4..column * 4 + 4];
                let (r, b) = if format == FrameFormat::RAWRGB {
                    (input[0], input[2])
                } else {
                    (input[2], input[0])
                };
                output.copy_from_slice(&[r, input[1], b, 255]);
            }
        }
        image::RgbaImage::from_raw(width as u32, height as u32, rgba)
            .ok_or_else(|| Error::Capture("RGB frame buffer size mismatch".to_owned()))
    }

    pub(super) fn yuyv_to_rgba(
        width: usize,
        height: usize,
        data: &[u8],
    ) -> Result<image::RgbaImage> {
        if !width.is_multiple_of(2) {
            return Err(Error::Capture("YUYV frame has an odd width".to_owned()));
        }
        let stride = row_stride(data.len(), height, width * 2)
            .ok_or_else(|| Error::Capture("YUYV frame has an inconsistent size".to_owned()))?;
        let mut rgba = vec![0_u8; width * height * 4];
        for row in 0..height {
            let source = &data[row * stride..row * stride + width * 2];
            let target = &mut rgba[row * width * 4..(row + 1) * width * 4];
            for pair in 0..width / 2 {
                let chunk = &source[pair * 4..pair * 4 + 4];
                let left = yuv_to_rgb(chunk[0], chunk[1], chunk[3]);
                let right = yuv_to_rgb(chunk[2], chunk[1], chunk[3]);
                target[pair * 8..pair * 8 + 4].copy_from_slice(&left);
                target[pair * 8 + 4..pair * 8 + 8].copy_from_slice(&right);
            }
        }
        image::RgbaImage::from_raw(width as u32, height as u32, rgba)
            .ok_or_else(|| Error::Capture("YUYV frame buffer size mismatch".to_owned()))
    }

    pub(super) fn nv12_to_rgba(
        width: usize,
        height: usize,
        data: &[u8],
    ) -> Result<image::RgbaImage> {
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(Error::Capture(
                "NV12 frame has empty or odd dimensions".to_owned(),
            ));
        }
        // NV12 is one luma plane followed by interleaved chroma; both planes
        // share a row stride.  The stride may be padded beyond the width.
        let total = (height * 3) / 2;
        if !data.len().is_multiple_of(total) {
            return Err(Error::Capture(
                "NV12 frame has an inconsistent size".to_owned(),
            ));
        }
        let stride = data.len() / total;
        if stride < width || data.len() < stride * total {
            return Err(Error::Capture(
                "NV12 frame has an inconsistent size".to_owned(),
            ));
        }
        let chroma_base = stride * height;
        let mut rgba = vec![0_u8; width * height * 4];
        for row in 0..height {
            let luma = &data[row * stride..row * stride + width];
            let chroma = &data[chroma_base + (row / 2) * stride..];
            let target = &mut rgba[row * width * 4..(row + 1) * width * 4];
            for column in 0..width {
                let pair = column / 2;
                let u = chroma[pair * 2];
                let v = chroma[pair * 2 + 1];
                target[column * 4..column * 4 + 4].copy_from_slice(&yuv_to_rgb(luma[column], u, v));
            }
        }
        image::RgbaImage::from_raw(width as u32, height as u32, rgba)
            .ok_or_else(|| Error::Capture("NV12 frame buffer size mismatch".to_owned()))
    }

    /// BT.601 studio-range YCbCr to RGBA, matching `nokhwa`'s conversion.
    fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 4] {
        let c = (i32::from(y) - 16) * 298;
        let d = i32::from(u) - 128;
        let e = i32::from(v) - 128;
        let r = clamp_byte((c + 409 * e + 128) >> 8);
        let g = clamp_byte((c - 100 * d - 208 * e + 128) >> 8);
        let b = clamp_byte((c + 516 * d + 128) >> 8);
        [r, g, b, 255]
    }

    fn clamp_byte(value: i32) -> u8 {
        value.clamp(0, 255) as u8
    }

    fn row_stride(data_len: usize, height: usize, minimum: usize) -> Option<usize> {
        if height == 0 || !data_len.is_multiple_of(height) {
            return None;
        }
        let stride = data_len / height;
        (stride >= minimum).then_some(stride)
    }
}

#[cfg(feature = "camera")]
use webcam::WebcamSource;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image(width: usize, height: usize) -> LoadedImage {
        let gray = GrayF32::filled(width, height, 0.4);
        LoadedImage {
            display: gray.display_rgba(),
            gray,
        }
    }

    #[test]
    fn image_file_source_yields_one_frame_then_ends() {
        let directory = std::env::temp_dir();
        let path = directory.join(format!("pm-capture-{}.png", std::process::id()));
        let gray = test_image(8, 6);
        gray.display.save(&path).expect("write test image");

        let mut source = ImageFileSource::open(&path).expect("open source");
        assert_eq!(source.description(), path.display().to_string());
        let frame = source
            .next_frame()
            .expect("first frame")
            .expect("a frame is available");
        assert_eq!(frame.sequence, 1);
        assert_eq!((frame.image.gray.width, frame.image.gray.height), (8, 6));
        assert!(source.next_frame().expect("exhausted").is_none());
        assert!(source.next_frame().expect("still exhausted").is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn malformed_capture_config_is_rejected() {
        let device = CaptureDevice {
            index: 0,
            name: "test".to_owned(),
            description: String::new(),
        };
        let error = open_webcam(
            &device,
            CaptureConfig {
                width: 0,
                height: 720,
                fps: 30,
            },
        )
        .err()
        .expect("zero width is invalid");
        assert!(matches!(error, Error::InvalidParameter(_)));
    }

    #[cfg(feature = "camera")]
    #[test]
    fn format_choice_prefers_the_requested_mode() {
        use nokhwa::utils::{CameraFormat, FrameFormat};
        let formats = [
            CameraFormat::new_from(640, 480, FrameFormat::YUYV, 30),
            CameraFormat::new_from(1280, 720, FrameFormat::MJPEG, 30),
            CameraFormat::new_from(1920, 1080, FrameFormat::MJPEG, 30),
            CameraFormat::new_from(1280, 720, FrameFormat::YUYV, 15),
        ];
        let chosen = webcam::choose_format(&formats, CaptureConfig::default())
            .expect("a supported format is available");
        assert_eq!(chosen.resolution().width_x, 1280);
        assert_eq!(chosen.resolution().height_y, 720);
        // The 30 fps mode wins over the same-resolution 15 fps mode.
        assert_eq!(chosen.frame_rate(), 30);
    }

    #[cfg(feature = "camera")]
    #[test]
    fn yuyv_conversion_produces_expected_luma() {
        // Two pixels: mid-grey luma with neutral chroma.
        let data = [128_u8, 128, 128, 128];
        let rgba = webcam::yuyv_to_rgba(2, 1, &data).expect("convert yuyv");
        assert_eq!(rgba.width(), 2);
        assert_eq!(rgba.height(), 1);
        for pixel in rgba.pixels() {
            assert!((120..=140).contains(&pixel[0]));
            assert_eq!(pixel[3], 255);
        }
    }

    #[cfg(feature = "camera")]
    #[test]
    fn nv12_conversion_produces_expected_luma() {
        // 2×2 luma plus one neutral chroma pair.
        let data = [128_u8, 128, 128, 128, 128, 128];
        let rgba = webcam::nv12_to_rgba(2, 2, &data).expect("convert nv12");
        assert_eq!((rgba.width(), rgba.height()), (2, 2));
        for pixel in rgba.pixels() {
            assert!((120..=140).contains(&pixel[0]));
        }
    }

    #[cfg(feature = "camera")]
    #[test]
    fn stride_padded_gray_is_supported() {
        // Two rows of width 3 padded to 4 bytes.
        let data = [10_u8, 20, 30, 0, 40, 50, 60, 0];
        let image = webcam::gray_to_loaded(3, 2, &data).expect("convert gray");
        assert_eq!(image.gray.width, 3);
        assert_eq!(image.gray.height, 2);
        assert!((image.gray.data[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((image.gray.data[5] - 60.0 / 255.0).abs() < 1e-6);
    }

    /// Hardware smoke test; run with
    /// `cargo test -p pm-core --features camera -- --ignored camera_opens`.
    #[cfg(feature = "camera")]
    #[test]
    #[ignore = "requires a connected camera"]
    fn camera_opens_and_delivers_a_frame() {
        let devices = list_cameras().expect("enumerate cameras");
        assert!(!devices.is_empty(), "no cameras found");
        println!(
            "camera: {} [{}]",
            devices[0].label(),
            devices[0].description
        );
        let mut source =
            open_webcam(&devices[0], CaptureConfig::default()).expect("open first camera");
        println!("stream: {}", source.description());
        let frame = source
            .next_frame()
            .expect("read a frame")
            .expect("source has a frame");
        assert!(frame.image.gray.width > 0);
        assert!(frame.image.gray.height > 0);
        println!(
            "frame {}: {}×{}",
            frame.sequence, frame.image.gray.width, frame.image.gray.height
        );
        source.stop().expect("stop stream");
    }
}
