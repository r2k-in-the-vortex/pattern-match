//! Model building, teaching transforms, and `.pmodel` persistence
//! (README §8).
//!
//! A model stores features in the model frame defined by the zero frame.  The
//! `.pmodel` container is a ZIP archive with a human-readable `model.json`,
//! a documented little-endian `features.bin`, an optional derived `mask.png`,
//! and an optional embedded `source.png`.

use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::descriptor::{
    CornerDescriptor, DESCRIPTOR_DIM, SegmentPairInvariant, describe_corners,
    segment_pair_invariants,
};
use crate::error::{Error, Result};
use crate::feature::{
    Corner, EdgeChain, EdgePoint, EdgeSet, FeatureParams, FeatureSet, Segment, extract_features,
    filter_features,
};
use crate::geometry::{RotatedRect, ZeroFrame};
use crate::gray::LoadedImage;
use crate::mask::{MaskBounds, MaskPrimitive, MaskRaster};

/// Manifest format identifier.
pub const MODEL_FORMAT: &str = "pattern-match-model";
/// Current model format version.
pub const MODEL_VERSION: u32 = 1;
/// Manifest file name inside the container.
pub const MANIFEST_FILE: &str = "model.json";
/// Binary feature file name inside the container.
pub const FEATURES_FILE: &str = "features.bin";
/// Derived mask image file name.
pub const MASK_FILE: &str = "mask.png";
/// Embedded 16-bit grayscale source file name.
pub const SOURCE_FILE: &str = "source.png";
/// Optional 8-bit RGBA display copy of the source.
pub const DISPLAY_FILE: &str = "display.png";

const FEATURE_MAGIC: &[u8; 8] = b"PMFEAT01";
const MAX_COUNT: usize = 5_000_000;
const MAX_SEGMENT_PAIRS: usize = 500;

/// Sources for coarse similarity hypotheses (README §9.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisSources {
    /// Oriented Hough voting only.
    Hough,
    /// Corner descriptors and segment pairs only.
    Descriptors,
    /// Both sources.
    Both,
}

/// Default run parameters stored with a model (README §8.2, Appendix A).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchDefaults {
    /// Minimum rotation in degrees.
    pub rotation_min_deg: f32,
    /// Maximum rotation in degrees.
    pub rotation_max_deg: f32,
    /// Minimum scale.
    pub scale_min: f32,
    /// Maximum scale.
    pub scale_max: f32,
    /// Maximum number of detections.
    pub max_detections: usize,
    /// Detection score threshold.
    pub score_threshold: f32,
    /// Minimum visible model weight.
    pub coverage_min: f32,
    /// Whether contrast polarity must agree.
    pub polarity_enabled: bool,
    /// Enabled hypothesis sources.
    pub hypothesis_sources: HypothesisSources,
    /// RANSAC confidence.
    pub ransac_confidence: f32,
    /// Deterministic RANSAC seed.
    pub ransac_seed: u64,
}

impl Default for SearchDefaults {
    fn default() -> Self {
        Self {
            rotation_min_deg: -180.0,
            rotation_max_deg: 180.0,
            scale_min: 0.5,
            scale_max: 2.0,
            max_detections: 10,
            score_threshold: 0.55,
            coverage_min: 0.4,
            polarity_enabled: true,
            hypothesis_sources: HypothesisSources::Both,
            ransac_confidence: 0.99,
            ransac_seed: 42,
        }
    }
}

/// Teaching inputs for [`Model::build`].
#[derive(Clone, Debug)]
pub struct TeachSettings {
    /// Model name.
    pub name: String,
    /// Rotated ROI in image coordinates.
    pub roi: RotatedRect,
    /// Zero frame in image coordinates.
    pub zero: ZeroFrame,
    /// Feature extraction parameters.
    pub params: FeatureParams,
    /// Mask primitives in model coordinates.
    pub mask: Vec<MaskPrimitive>,
    /// Default run parameters.
    pub search_defaults: SearchDefaults,
}

/// A taught model with features in model coordinates.
#[derive(Clone, Debug)]
pub struct Model {
    /// Model name.
    pub name: String,
    /// Creation time as Unix seconds.
    pub created_unix_seconds: u64,
    /// Taught ROI in image coordinates.
    pub roi: RotatedRect,
    /// Taught zero frame in image coordinates.
    pub zero: ZeroFrame,
    /// Extraction parameters used at teach time.
    pub params: FeatureParams,
    /// Level-0 features in model coordinates.
    pub features: FeatureSet,
    /// Corner descriptors, parallel to `features.corners`.
    pub descriptors: Vec<CornerDescriptor>,
    /// Segment-pair invariants.
    pub segment_pairs: Vec<SegmentPairInvariant>,
    /// Mask primitives in model coordinates.
    pub mask_primitives: Vec<MaskPrimitive>,
    /// Bounds of the derived mask raster.
    pub mask_bounds: MaskBounds,
    /// Mask raster resolution in model units per cell.
    pub mask_resolution: f32,
    /// Derived exclusion raster.
    pub mask: MaskRaster,
    /// Sum of normalized edge weights removed by the mask.
    pub excluded_edge_weight: f32,
    /// Fraction of ROI edge weight removed by the mask.
    pub excluded_edge_weight_fraction: f32,
    /// Default run parameters.
    pub search_defaults: SearchDefaults,
    /// Embedded source image, when present.
    pub source: Option<LoadedImage>,
    /// Source image dimensions.
    pub source_size: (usize, usize),
}

impl Model {
    /// Build a model from a source image and teach settings.
    ///
    /// Features are extracted at full resolution, clipped to the ROI, converted
    /// to the model frame, weighted, filtered by the mask, and finally described.
    pub fn build(source: &LoadedImage, settings: &TeachSettings) -> Result<Self> {
        if source.gray.is_empty() {
            return Err(Error::InvalidParameter("source image is empty".to_owned()));
        }
        if !settings.roi.is_valid() {
            return Err(Error::InvalidParameter(
                "ROI must have finite center/angle and positive dimensions".to_owned(),
            ));
        }
        if !settings.zero.is_valid() {
            return Err(Error::InvalidParameter(
                "zero frame must be finite".to_owned(),
            ));
        }
        if !settings.params.blur_sigma.is_finite() || settings.params.blur_sigma < 0.0 {
            return Err(Error::InvalidParameter(
                "blur sigma must be finite and non-negative".to_owned(),
            ));
        }

        let extracted = extract_features(&source.gray, &settings.params);
        let mut clipped = filter_features(&extracted, |x, y| settings.roi.contains(x, y));
        normalize_weights(&mut clipped);
        let model_features = features_to_model(&clipped, &settings.zero);

        let mask_bounds = mask_bounds_for(&settings.roi, &settings.zero, &settings.mask);
        let mask = MaskRaster::rasterize(&settings.mask, mask_bounds, 1.0);
        let mask_resolution = mask.resolution;

        let total_edge_weight: f32 = model_features
            .edges
            .points
            .iter()
            .map(|point| point.weight)
            .sum();
        let features = filter_features(&model_features, |x, y| !mask.is_excluded([x, y]));
        let kept_edge_weight: f32 = features.edges.points.iter().map(|point| point.weight).sum();
        let excluded_edge_weight = (total_edge_weight - kept_edge_weight).max(0.0);
        let excluded_edge_weight_fraction = if total_edge_weight > 0.0 {
            excluded_edge_weight / total_edge_weight
        } else {
            0.0
        };

        let descriptors = describe_corners(&features.corners, &features.edges);
        let segment_pairs = segment_pair_invariants(&features.segments, MAX_SEGMENT_PAIRS);
        // Remove any scale drift introduced by extracting descriptors after
        // filtering; the values are already normalized, so no further step.

        Ok(Self {
            name: if settings.name.trim().is_empty() {
                "model".to_owned()
            } else {
                settings.name.clone()
            },
            created_unix_seconds: unix_now(),
            roi: settings.roi,
            zero: settings.zero,
            params: settings.params,
            features,
            descriptors,
            segment_pairs,
            mask_primitives: settings.mask.clone(),
            mask_bounds,
            mask_resolution,
            mask,
            excluded_edge_weight,
            excluded_edge_weight_fraction,
            search_defaults: settings.search_defaults,
            source: Some(source.clone()),
            source_size: (source.gray.width, source.gray.height),
        })
    }

    /// Feature counts as `(edges, chains, corners, segments, descriptors, pairs)`.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.features.edge_count(),
            self.features.chain_count(),
            self.features.corner_count(),
            self.features.segment_count(),
            self.descriptors.len(),
            self.segment_pairs.len(),
        )
    }

    /// Model features converted to level-`level` coordinates (`model / 2^L`).
    ///
    /// Level 0 returns the stored features.  Normals and orientations are
    /// unchanged; lengths and positions scale by `2^-L`.
    #[must_use]
    pub fn features_at_level(&self, level: usize) -> FeatureSet {
        if level == 0 {
            return self.features.clone();
        }
        let factor = (1u32 << level) as f32;
        let inv = 1.0 / factor;
        let points = self
            .features
            .edges
            .points
            .iter()
            .map(|point| EdgePoint {
                x: point.x * inv,
                y: point.y * inv,
                ..*point
            })
            .collect();
        let corners = self
            .features
            .corners
            .iter()
            .map(|corner| Corner {
                x: corner.x * inv,
                y: corner.y * inv,
                ..*corner
            })
            .collect();
        let segments = self
            .features
            .segments
            .iter()
            .map(|segment| Segment {
                x0: segment.x0 * inv,
                y0: segment.y0 * inv,
                x1: segment.x1 * inv,
                y1: segment.y1 * inv,
                length: segment.length * inv,
                ..*segment
            })
            .collect();
        FeatureSet {
            edges: EdgeSet {
                points,
                chains: self.features.edges.chains.clone(),
            },
            corners,
            segments,
        }
    }

    /// Mask raster converted to level-`level` coordinates.
    #[must_use]
    pub fn mask_at_level(&self, level: usize) -> MaskRaster {
        if level == 0 {
            return self.mask.clone();
        }
        let factor = (1u32 << level) as f32;
        let inv = 1.0 / factor;
        let primitives: Vec<MaskPrimitive> = self
            .mask_primitives
            .iter()
            .map(|primitive| primitive.scaled(inv))
            .collect();
        let bounds = MaskBounds {
            min_x: self.mask_bounds.min_x * inv,
            min_y: self.mask_bounds.min_y * inv,
            max_x: self.mask_bounds.max_x * inv,
            max_y: self.mask_bounds.max_y * inv,
        };
        MaskRaster::rasterize(&primitives, bounds, self.mask.resolution * inv)
    }

    /// Convert the stored model-frame features back to image coordinates for
    /// display in the teach view.
    #[must_use]
    pub fn features_in_image_coords(&self) -> FeatureSet {
        let zero = self.zero;
        let points: Vec<EdgePoint> = self
            .features
            .edges
            .points
            .iter()
            .map(|point| {
                let p = zero.to_image([point.x, point.y]);
                let n = zero.vector_to_image([point.nx, point.ny]);
                EdgePoint {
                    x: p[0],
                    y: p[1],
                    nx: n[0],
                    ny: n[1],
                    ..*point
                }
            })
            .collect();
        let corners: Vec<Corner> = self
            .features
            .corners
            .iter()
            .map(|corner| {
                let p = zero.to_image([corner.x, corner.y]);
                Corner {
                    x: p[0],
                    y: p[1],
                    orientation: zero
                        .angle_to_image(corner.orientation)
                        .rem_euclid(std::f32::consts::PI),
                    ..*corner
                }
            })
            .collect();
        let segments: Vec<Segment> = self
            .features
            .segments
            .iter()
            .map(|segment| {
                let a = zero.to_image([segment.x0, segment.y0]);
                let b = zero.to_image([segment.x1, segment.y1]);
                let length = (b[0] - a[0]).hypot(b[1] - a[1]);
                let mut orientation = (b[1] - a[1]).atan2(b[0] - a[0]);
                if orientation < 0.0 {
                    orientation += std::f32::consts::PI;
                }
                if orientation >= std::f32::consts::PI {
                    orientation -= std::f32::consts::PI;
                }
                Segment {
                    x0: a[0],
                    y0: a[1],
                    x1: b[0],
                    y1: b[1],
                    length,
                    orientation,
                    ..*segment
                }
            })
            .collect();
        FeatureSet {
            edges: EdgeSet {
                points,
                chains: self.features.edges.chains.clone(),
            },
            corners,
            segments,
        }
    }

    /// Mask primitives converted to image coordinates for display.
    #[must_use]
    pub fn mask_primitives_in_image_coords(&self) -> Vec<MaskPrimitive> {
        self.mask_primitives
            .iter()
            .map(|primitive| primitive.to_image(&self.zero))
            .collect()
    }

    /// Save the model as a `.pmodel` ZIP container.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let manifest = ModelManifest {
            format: MODEL_FORMAT.to_owned(),
            version: MODEL_VERSION,
            name: self.name.clone(),
            created_unix_seconds: self.created_unix_seconds,
            source: SourceMeta {
                width: self.source_size.0,
                height: self.source_size.1,
                embedded: self.source.is_some(),
            },
            roi: self.roi,
            zero: self.zero,
            params: self.params,
            mask: MaskMeta {
                primitives: self.mask_primitives.clone(),
                bounds: self.mask_bounds,
                resolution: self.mask_resolution,
                mask_hash: self.mask.hash(),
                excluded_edge_weight: self.excluded_edge_weight,
                excluded_edge_weight_fraction: self.excluded_edge_weight_fraction,
            },
            search_defaults: self.search_defaults,
            counts: Counts {
                edges: self.features.edge_count(),
                chains: self.features.chain_count(),
                corners: self.features.corner_count(),
                segments: self.features.segment_count(),
                descriptors: self.descriptors.len(),
                segment_pairs: self.segment_pairs.len(),
            },
        };
        let manifest_json = serde_json::to_vec_pretty(&manifest)
            .map_err(|err| Error::Format(format!("serialize manifest: {err}")))?;
        let feature_bytes =
            encode_features(&self.features, &self.descriptors, &self.segment_pairs)?;

        let file = File::create(path)?;
        let mut zip = zip::ZipWriter::new(file);
        let deflated = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        let stored = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);

        zip.start_file(MANIFEST_FILE, deflated).map_err(zip_error)?;
        zip.write_all(&manifest_json)?;
        zip.start_file(FEATURES_FILE, deflated).map_err(zip_error)?;
        zip.write_all(&feature_bytes)?;
        zip.start_file(MASK_FILE, stored).map_err(zip_error)?;
        zip.write_all(&png_bytes(&self.mask.to_rgba())?)?;
        if let Some(source) = &self.source {
            zip.start_file(SOURCE_FILE, stored).map_err(zip_error)?;
            zip.write_all(&gray_png_bytes(&source.gray)?)?;
            zip.start_file(DISPLAY_FILE, stored).map_err(zip_error)?;
            zip.write_all(&png_bytes(&source.display)?)?;
        }
        zip.finish().map_err(zip_error)?;
        Ok(())
    }

    /// Load a `.pmodel` container, validating the manifest and feature arrays.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mut archive = zip::ZipArchive::new(file).map_err(zip_error)?;
        let manifest_bytes = read_zip_file(&mut archive, MANIFEST_FILE)?;
        let manifest: ModelManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|err| Error::Format(format!("manifest JSON: {err}")))?;
        if manifest.format != MODEL_FORMAT {
            return Err(Error::Format(format!(
                "unexpected model format {:?}",
                manifest.format
            )));
        }
        if manifest.version != MODEL_VERSION {
            return Err(Error::Format(format!(
                "unsupported model version {}",
                manifest.version
            )));
        }
        if !manifest.roi.is_valid() {
            return Err(Error::Format("manifest ROI is invalid".to_owned()));
        }
        if !manifest.zero.is_valid() {
            return Err(Error::Format("manifest zero frame is invalid".to_owned()));
        }
        if !manifest.mask.bounds.is_valid()
            || !manifest.mask.resolution.is_finite()
            || manifest.mask.resolution <= 0.0
        {
            return Err(Error::Format("manifest mask bounds are invalid".to_owned()));
        }

        let feature_bytes = read_zip_file(&mut archive, FEATURES_FILE)?;
        let decoded = decode_features(&feature_bytes)?;
        let expected = (
            manifest.counts.edges,
            manifest.counts.chains,
            manifest.counts.corners,
            manifest.counts.segments,
            manifest.counts.descriptors,
            manifest.counts.segment_pairs,
        );
        let actual = (
            decoded.features.edge_count(),
            decoded.features.chain_count(),
            decoded.features.corner_count(),
            decoded.features.segment_count(),
            decoded.descriptors.len(),
            decoded.segment_pairs.len(),
        );
        if expected != actual {
            return Err(Error::Format(format!(
                "manifest counts {expected:?} do not match features.bin {actual:?}"
            )));
        }

        let source = if manifest.source.embedded {
            let bytes = read_zip_file(&mut archive, SOURCE_FILE)?;
            let decoded = image::load_from_memory(&bytes)
                .map_err(|err| Error::Format(format!("source image: {err}")))?;
            let gray = crate::gray::GrayF32::from_dynamic_image(&decoded)
                .map_err(|err| Error::Format(format!("source image: {err}")))?;
            if (gray.width, gray.height) != (manifest.source.width, manifest.source.height) {
                return Err(Error::Format(
                    "embedded source dimensions do not match the manifest".to_owned(),
                ));
            }
            let display = match read_zip_file(&mut archive, DISPLAY_FILE) {
                Ok(display_bytes) => image::load_from_memory(&display_bytes)
                    .map_err(|err| Error::Format(format!("display image: {err}")))?
                    .to_rgba8(),
                Err(_) => gray.display_rgba(),
            };
            if display.dimensions() != (gray.width as u32, gray.height as u32) {
                return Err(Error::Format(
                    "display image dimensions do not match the source".to_owned(),
                ));
            }
            Some(LoadedImage { gray, display })
        } else {
            None
        };

        let mask = MaskRaster::rasterize(
            &manifest.mask.primitives,
            manifest.mask.bounds,
            manifest.mask.resolution,
        );
        if mask.hash() != manifest.mask.mask_hash {
            return Err(Error::Format(
                "mask raster hash does not match the stored primitives".to_owned(),
            ));
        }
        Ok(Self {
            name: manifest.name,
            created_unix_seconds: manifest.created_unix_seconds,
            roi: manifest.roi,
            zero: manifest.zero,
            params: manifest.params,
            features: decoded.features,
            descriptors: decoded.descriptors,
            segment_pairs: decoded.segment_pairs,
            mask_primitives: manifest.mask.primitives,
            mask_bounds: manifest.mask.bounds,
            mask_resolution: mask.resolution,
            mask,
            excluded_edge_weight: manifest.mask.excluded_edge_weight,
            excluded_edge_weight_fraction: manifest.mask.excluded_edge_weight_fraction,
            search_defaults: manifest.search_defaults,
            source,
            source_size: (manifest.source.width, manifest.source.height),
        })
    }
}

/// Mask bounds covering the ROI and every primitive, with room for brushes.
///
/// Coordinates are converted to the model frame using `zero`.
#[must_use]
pub fn mask_bounds_for(
    roi: &RotatedRect,
    zero: &ZeroFrame,
    primitives: &[MaskPrimitive],
) -> MaskBounds {
    let mut max_radius = 0.0f32;
    for primitive in primitives {
        if let MaskPrimitive::Brush { radius, .. } = primitive {
            max_radius = max_radius.max(*radius);
        }
    }
    let roi_in_model: Vec<[f32; 2]> = roi.corners().iter().map(|&p| zero.to_model(p)).collect();
    let mut bounds = MaskBounds::from_points(roi_in_model, max_radius.max(0.0) + 2.0);
    for primitive in primitives {
        if let Some(primitive_bounds) = primitive.bounds() {
            bounds = bounds.union(primitive_bounds);
        }
    }
    bounds
}

fn normalize_weights(features: &mut FeatureSet) {
    let max_edge = features
        .edges
        .points
        .iter()
        .map(|point| point.magnitude)
        .fold(0.0f32, f32::max);
    if max_edge > 0.0 {
        for point in &mut features.edges.points {
            point.weight = (point.magnitude / max_edge).clamp(0.05, 1.0);
        }
    }
    let max_corner = features
        .corners
        .iter()
        .map(|corner| corner.response)
        .fold(0.0f32, f32::max);
    if max_corner > 0.0 {
        for corner in &mut features.corners {
            corner.weight = (corner.response / max_corner).clamp(0.05, 1.0);
        }
    }
    let max_segment = features
        .segments
        .iter()
        .map(|segment| segment.length)
        .fold(0.0f32, f32::max);
    if max_segment > 0.0 {
        for segment in &mut features.segments {
            segment.weight = (segment.length / max_segment).clamp(0.05, 1.0);
        }
    }
}

fn features_to_model(features: &FeatureSet, zero: &ZeroFrame) -> FeatureSet {
    let points = features
        .edges
        .points
        .iter()
        .map(|point| {
            let p = zero.to_model([point.x, point.y]);
            let mut n = zero.vector_to_model([point.nx, point.ny]);
            let norm = n[0].hypot(n[1]);
            if norm > 0.0 {
                n = [n[0] / norm, n[1] / norm];
            }
            EdgePoint {
                x: p[0],
                y: p[1],
                nx: n[0],
                ny: n[1],
                ..*point
            }
        })
        .collect();
    let corners = features
        .corners
        .iter()
        .map(|corner| {
            let p = zero.to_model([corner.x, corner.y]);
            Corner {
                x: p[0],
                y: p[1],
                orientation: zero
                    .angle_to_model(corner.orientation)
                    .rem_euclid(std::f32::consts::PI),
                ..*corner
            }
        })
        .collect();
    let segments = features
        .segments
        .iter()
        .map(|segment| {
            let a = zero.to_model([segment.x0, segment.y0]);
            let b = zero.to_model([segment.x1, segment.y1]);
            make_segment(a, b, segment.weight)
        })
        .collect();
    FeatureSet {
        edges: EdgeSet {
            points,
            chains: features.edges.chains.clone(),
        },
        corners,
        segments,
    }
}

fn make_segment(a: [f32; 2], b: [f32; 2], weight: f32) -> Segment {
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    let mut orientation = (b[1] - a[1]).atan2(b[0] - a[0]);
    if orientation < 0.0 {
        orientation += std::f32::consts::PI;
    }
    if orientation >= std::f32::consts::PI {
        orientation -= std::f32::consts::PI;
    }
    Segment {
        x0: a[0],
        y0: a[1],
        x1: b[0],
        y1: b[1],
        length,
        orientation,
        weight,
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn zip_error(error: zip::result::ZipError) -> Error {
    Error::Format(format!("zip: {error}"))
}

fn read_zip_file(archive: &mut zip::ZipArchive<File>, name: &str) -> Result<Vec<u8>> {
    let mut file = archive
        .by_name(name)
        .map_err(|error| Error::Format(format!("missing {name}: {error}")))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn png_bytes(image: &image::RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|err| Error::Format(format!("encode PNG: {err}")))?;
    Ok(bytes)
}

fn gray_png_bytes(gray: &crate::gray::GrayF32) -> Result<Vec<u8>> {
    let mut image = image::ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(
        gray.width as u32,
        gray.height as u32,
    );
    for (pixel, &value) in image.pixels_mut().zip(&gray.data) {
        *pixel = image::Luma([(value.clamp(0.0, 1.0) * 65535.0).round() as u16]);
    }
    let mut bytes = Vec::new();
    image::DynamicImage::ImageLuma16(image)
        .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|err| Error::Format(format!("encode grayscale PNG: {err}")))?;
    Ok(bytes)
}

fn encode_features(
    features: &FeatureSet,
    descriptors: &[CornerDescriptor],
    pairs: &[SegmentPairInvariant],
) -> Result<Vec<u8>> {
    if descriptors.len() != features.corners.len() {
        return Err(Error::InvalidParameter(
            "descriptor count does not match corner count".to_owned(),
        ));
    }
    for descriptor in descriptors {
        if descriptor.values.len() != DESCRIPTOR_DIM {
            return Err(Error::InvalidParameter(format!(
                "descriptor has {} values, expected {DESCRIPTOR_DIM}",
                descriptor.values.len()
            )));
        }
    }
    let counts = [
        features.edges.points.len(),
        features.edges.chains.len(),
        features.corners.len(),
        descriptors.len(),
        features.segments.len(),
        pairs.len(),
    ];
    if counts.iter().any(|&count| count > MAX_COUNT) {
        return Err(Error::InvalidParameter(
            "feature count exceeds the model format limit".to_owned(),
        ));
    }

    let mut out = Vec::new();
    out.extend_from_slice(FEATURE_MAGIC);
    out.extend_from_slice(&MODEL_VERSION.to_le_bytes());
    put_count(&mut out, features.edges.points.len());
    put_count(&mut out, features.edges.chains.len());
    put_count(&mut out, features.corners.len());
    put_count(&mut out, descriptors.len());
    put_count(&mut out, features.segments.len());
    put_count(&mut out, pairs.len());
    put_count(&mut out, DESCRIPTOR_DIM);

    for point in &features.edges.points {
        put_f32(&mut out, point.x);
        put_f32(&mut out, point.y);
        put_f32(&mut out, point.nx);
        put_f32(&mut out, point.ny);
        put_f32(&mut out, point.magnitude);
        put_f32(&mut out, point.weight);
        out.push(point.sign as u8);
        out.extend_from_slice(&point.chain_id.to_le_bytes());
    }
    for chain in &features.edges.chains {
        out.extend_from_slice(&chain.id.to_le_bytes());
        put_count(&mut out, chain.points.len());
        for &index in &chain.points {
            out.extend_from_slice(&(index as u32).to_le_bytes());
        }
    }
    for corner in &features.corners {
        put_f32(&mut out, corner.x);
        put_f32(&mut out, corner.y);
        put_f32(&mut out, corner.response);
        put_f32(&mut out, corner.orientation);
        put_f32(&mut out, corner.weight);
    }
    for descriptor in descriptors {
        out.extend_from_slice(&(descriptor.corner_index as u32).to_le_bytes());
        for &value in &descriptor.values {
            put_f32(&mut out, value);
        }
    }
    for segment in &features.segments {
        put_f32(&mut out, segment.x0);
        put_f32(&mut out, segment.y0);
        put_f32(&mut out, segment.x1);
        put_f32(&mut out, segment.y1);
        put_f32(&mut out, segment.length);
        put_f32(&mut out, segment.orientation);
        put_f32(&mut out, segment.weight);
    }
    for pair in pairs {
        out.extend_from_slice(&(pair.first as u32).to_le_bytes());
        out.extend_from_slice(&(pair.second as u32).to_le_bytes());
        put_f32(&mut out, pair.length_ratio);
        put_f32(&mut out, pair.angle_between);
        put_f32(&mut out, pair.midpoint_distance);
        put_f32(&mut out, pair.midpoint_angle);
    }
    Ok(out)
}

struct DecodedFeatures {
    features: FeatureSet,
    descriptors: Vec<CornerDescriptor>,
    segment_pairs: Vec<SegmentPairInvariant>,
}

fn decode_features(bytes: &[u8]) -> Result<DecodedFeatures> {
    let mut reader = ByteReader::new(bytes);
    if reader.take(8)? != FEATURE_MAGIC {
        return Err(Error::Format("features.bin magic mismatch".to_owned()));
    }
    let version = reader.u32()?;
    if version != MODEL_VERSION {
        return Err(Error::Format(format!(
            "unsupported features.bin version {version}"
        )));
    }
    let edge_count = reader.count()?;
    let chain_count = reader.count()?;
    let corner_count = reader.count()?;
    let descriptor_count = reader.count()?;
    let segment_count = reader.count()?;
    let pair_count = reader.count()?;
    let descriptor_dim = reader.count()?;
    if descriptor_dim != DESCRIPTOR_DIM {
        return Err(Error::Format(format!(
            "descriptor dimension {descriptor_dim}, expected {DESCRIPTOR_DIM}"
        )));
    }
    if descriptor_count != corner_count {
        return Err(Error::Format(
            "descriptor count must equal corner count".to_owned(),
        ));
    }

    let mut points = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let x = reader.f32()?;
        let y = reader.f32()?;
        let nx = reader.f32()?;
        let ny = reader.f32()?;
        let magnitude = reader.f32()?;
        let weight = reader.f32()?;
        let sign = reader.i8()?;
        let chain_id = reader.u32()?;
        if sign != 1 && sign != -1 {
            return Err(Error::Format(format!("invalid edge sign {sign}")));
        }
        points.push(EdgePoint {
            x,
            y,
            nx,
            ny,
            magnitude,
            sign,
            chain_id,
            weight,
        });
    }

    let mut chains = Vec::with_capacity(chain_count);
    for _ in 0..chain_count {
        let id = reader.u32()?;
        let count = reader.count()?;
        if count > edge_count {
            return Err(Error::Format(
                "chain index count exceeds edge count".to_owned(),
            ));
        }
        let mut indices = Vec::with_capacity(count);
        for _ in 0..count {
            let index = reader.u32()? as usize;
            if index >= edge_count {
                return Err(Error::Format(format!("chain index {index} out of range")));
            }
            indices.push(index);
        }
        chains.push(EdgeChain {
            id,
            points: indices,
        });
    }

    let mut corners = Vec::with_capacity(corner_count);
    for _ in 0..corner_count {
        corners.push(Corner {
            x: reader.f32()?,
            y: reader.f32()?,
            response: reader.f32()?,
            orientation: reader.f32()?,
            weight: reader.f32()?,
        });
    }

    let mut descriptors = Vec::with_capacity(descriptor_count);
    for _ in 0..descriptor_count {
        let corner_index = reader.u32()? as usize;
        if corner_index >= corner_count {
            return Err(Error::Format(format!(
                "descriptor corner index {corner_index} out of range"
            )));
        }
        let mut values = Vec::with_capacity(DESCRIPTOR_DIM);
        for _ in 0..DESCRIPTOR_DIM {
            values.push(reader.f32()?);
        }
        descriptors.push(CornerDescriptor {
            corner_index,
            values,
        });
    }

    let mut segments = Vec::with_capacity(segment_count);
    for _ in 0..segment_count {
        let x0 = reader.f32()?;
        let y0 = reader.f32()?;
        let x1 = reader.f32()?;
        let y1 = reader.f32()?;
        let length = reader.f32()?;
        let orientation = reader.f32()?;
        let weight = reader.f32()?;
        if length < 0.0 {
            return Err(Error::Format("negative segment length".to_owned()));
        }
        segments.push(Segment {
            x0,
            y0,
            x1,
            y1,
            length,
            orientation,
            weight,
        });
    }

    let mut segment_pairs = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        let first = reader.u32()? as usize;
        let second = reader.u32()? as usize;
        if first >= segment_count || second >= segment_count {
            return Err(Error::Format("segment-pair index out of range".to_owned()));
        }
        segment_pairs.push(SegmentPairInvariant {
            first,
            second,
            length_ratio: reader.f32()?,
            angle_between: reader.f32()?,
            midpoint_distance: reader.f32()?,
            midpoint_angle: reader.f32()?,
        });
    }

    Ok(DecodedFeatures {
        features: FeatureSet {
            edges: EdgeSet { points, chains },
            corners,
            segments,
        },
        descriptors,
        segment_pairs,
    })
}

fn put_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_count(out: &mut Vec<u8>, value: usize) {
    out.extend_from_slice(&(value as u32).to_le_bytes());
}

struct ByteReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self.position + length;
        if end > self.data.len() {
            return Err(Error::Format(format!(
                "features.bin truncated at byte {} (needed {length})",
                self.position
            )));
        }
        let slice = &self.data[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    }

    fn count(&mut self) -> Result<usize> {
        let count = self.u32()? as usize;
        if count > MAX_COUNT {
            return Err(Error::Format(format!("unreasonable count {count}")));
        }
        Ok(count)
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }

    fn f32(&mut self) -> Result<f32> {
        let bytes = self.take(4)?;
        let value = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
        if !value.is_finite() {
            return Err(Error::Format("non-finite value in features.bin".to_owned()));
        }
        Ok(value)
    }
}

#[derive(Serialize, Deserialize)]
struct ModelManifest {
    format: String,
    version: u32,
    name: String,
    created_unix_seconds: u64,
    source: SourceMeta,
    roi: RotatedRect,
    zero: ZeroFrame,
    params: FeatureParams,
    mask: MaskMeta,
    search_defaults: SearchDefaults,
    counts: Counts,
}

#[derive(Serialize, Deserialize)]
struct SourceMeta {
    width: usize,
    height: usize,
    embedded: bool,
}

#[derive(Serialize, Deserialize)]
struct MaskMeta {
    primitives: Vec<MaskPrimitive>,
    bounds: MaskBounds,
    resolution: f32,
    mask_hash: u64,
    excluded_edge_weight: f32,
    excluded_edge_weight_fraction: f32,
}

#[derive(Serialize, Deserialize)]
struct Counts {
    edges: usize,
    chains: usize,
    corners: usize,
    segments: usize,
    descriptors: usize,
    segment_pairs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gray::GrayF32;
    use crate::mask::MaskPrimitive;

    fn source_image() -> LoadedImage {
        let width = 80;
        let height = 64;
        let mut data = vec![0.1f32; width * height];
        for y in 12..52 {
            for x in 14..66 {
                data[y * width + x] = 0.9;
            }
        }
        let gray = GrayF32::new(width, height, data).unwrap();
        let display = gray.display_rgba();
        LoadedImage { gray, display }
    }

    fn settings() -> TeachSettings {
        TeachSettings {
            name: "test-model".to_owned(),
            roi: RotatedRect::new(40.0, 32.0, 56.0, 44.0, 0.0),
            zero: ZeroFrame {
                origin: [40.0, 32.0],
                angle: 0.2,
            },
            params: FeatureParams::default(),
            mask: vec![MaskPrimitive::Polygon {
                // Covers part of the top rectangle edge in model coordinates.
                points: vec![[-10.0, -23.0], [2.0, -23.0], [2.0, -16.0], [-10.0, -16.0]],
                add: true,
            }],
            search_defaults: SearchDefaults::default(),
        }
    }

    #[test]
    fn model_build_clips_to_roi_and_masks() {
        let source = source_image();
        let model = Model::build(&source, &settings()).unwrap();
        assert!(model.features.edge_count() > 0);
        assert!(model.features.corner_count() >= 4);
        assert!(model.features.segment_count() >= 4);
        assert_eq!(model.descriptors.len(), model.features.corner_count());
        assert_eq!(DESCRIPTOR_DIM, model.descriptors[0].values.len());
        assert!(model.excluded_edge_weight_fraction > 0.0);
        assert!(!model.segment_pairs.is_empty());
        assert_eq!(model.source_size, (80, 64));
        assert!(model.mask.excluded_count() > 0);
        // Model coordinates are in the zero frame: the ROI center maps to 0.
        let center = model.zero.to_model([40.0, 32.0]);
        assert!(center[0].abs() < 1e-5 && center[1].abs() < 1e-5);
    }

    #[test]
    fn model_roundtrip_preserves_features_descriptors_and_mask() {
        let source = source_image();
        let model = Model::build(&source, &settings()).unwrap();
        let path =
            std::env::temp_dir().join(format!("pm-model-test-{}.pmodel", std::process::id()));
        model.save(&path).unwrap();
        let loaded = Model::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.name, model.name);
        assert_eq!(loaded.roi, model.roi);
        assert_eq!(loaded.zero, model.zero);
        assert_eq!(loaded.params, model.params);
        assert_eq!(loaded.features, model.features);
        assert_eq!(loaded.descriptors, model.descriptors);
        assert_eq!(loaded.segment_pairs, model.segment_pairs);
        assert_eq!(loaded.mask_primitives, model.mask_primitives);
        assert_eq!(loaded.mask.excluded, model.mask.excluded);
        assert_eq!(loaded.mask_bounds, model.mask_bounds);
        assert_eq!(loaded.mask.hash(), model.mask.hash());
        assert_eq!(loaded.search_defaults, model.search_defaults);
        assert_eq!(loaded.excluded_edge_weight, model.excluded_edge_weight);
        assert_eq!(
            loaded.excluded_edge_weight_fraction,
            model.excluded_edge_weight_fraction
        );
        assert!(loaded.source.is_some());
        let loaded_source = loaded.source.unwrap();
        assert_eq!(loaded_source.display, source.display);
        for (a, b) in loaded_source.gray.data.iter().zip(&source.gray.data) {
            assert!((a - b).abs() < 1e-4, "embedded grayscale drift {a} vs {b}");
        }
    }

    #[test]
    fn level_scaling_preserves_counts_and_mask() {
        let source = source_image();
        let model = Model::build(&source, &settings()).unwrap();
        let level1 = model.features_at_level(1);
        assert_eq!(level1.edge_count(), model.features.edge_count());
        assert_eq!(level1.corner_count(), model.features.corner_count());
        assert_eq!(level1.segment_count(), model.features.segment_count());
        if let (Some(a), Some(b)) = (
            model.features.edges.points.first(),
            level1.edges.points.first(),
        ) {
            assert!((a.x * 0.5 - b.x).abs() < 1e-5);
            assert!((a.y * 0.5 - b.y).abs() < 1e-5);
        }
        if let (Some(a), Some(b)) = (model.features.segments.first(), level1.segments.first()) {
            assert!((a.length * 0.5 - b.length).abs() < 1e-5);
            assert!((a.orientation - b.orientation).abs() < 1e-6);
        }

        let mask1 = model.mask_at_level(1);
        let mut checked = false;
        for y in 0..model.mask.height {
            for x in 0..model.mask.width {
                if model.mask.excluded[y * model.mask.width + x] {
                    let center = model.mask.cell_center(x, y);
                    assert!(
                        mask1.is_excluded([center[0] * 0.5, center[1] * 0.5]),
                        "level mask mismatch at cell ({x},{y})"
                    );
                    checked = true;
                }
            }
        }
        assert!(checked, "test model has excluded cells");
    }

    #[test]
    fn rejects_unknown_manifest_version() {
        let source = source_image();
        let model = Model::build(&source, &settings()).unwrap();
        let path = std::env::temp_dir().join(format!(
            "pm-model-version-test-{}.pmodel",
            std::process::id()
        ));
        model.save(&path).unwrap();

        // Rewrite the manifest with version 99.
        let file = File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let manifest_bytes = read_zip_file(&mut archive, MANIFEST_FILE).unwrap();
        let mut manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
        manifest["version"] = serde_json::Value::from(99);
        drop(archive);
        let out = File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(out);
        let opts = zip::write::SimpleFileOptions::default();
        zip.start_file(MANIFEST_FILE, opts).unwrap();
        zip.write_all(&serde_json::to_vec(&manifest).unwrap())
            .unwrap();
        zip.finish().unwrap();

        let error = Model::load(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(error, Error::Format(_)));
    }

    #[test]
    fn rejects_malformed_feature_payloads() {
        assert!(decode_features(b"short").is_err());
        let mut bytes = FEATURE_MAGIC.to_vec();
        bytes.extend_from_slice(&MODEL_VERSION.to_le_bytes());
        assert!(decode_features(&bytes).is_err(), "missing counts");
        bytes.extend_from_slice(&[0u8; 7 * 4]);
        assert!(
            decode_features(&bytes).is_err(),
            "zero descriptor dimension must be rejected"
        );
    }

    #[test]
    fn rejects_missing_manifest() {
        let path =
            std::env::temp_dir().join(format!("pm-model-empty-test-{}.pmodel", std::process::id()));
        let out = File::create(&path).unwrap();
        let zip = zip::ZipWriter::new(out);
        zip.finish().unwrap();
        let error = Model::load(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(matches!(error, Error::Format(_)));
    }

    #[test]
    fn invalid_teach_settings_are_rejected() {
        let source = source_image();
        let mut bad = settings();
        bad.roi.width = 0.0;
        assert!(Model::build(&source, &bad).is_err());
        let mut bad = settings();
        bad.zero.angle = f32::NAN;
        assert!(Model::build(&source, &bad).is_err());
    }
}
