//! Versioned detection-result JSON (README §12.3).
//!
//! [`ResultsFile`] is the serialized schema emitted by the CLI and UI.  It
//! wraps the detections with the model name, image dimensions, the parameters
//! used, and stage timings.  The internal [`Detection`](crate::score::Detection)
//! type is not serialized directly: the schema reports the pose as
//! `x, y, angle_deg, scale`, while the internal representation stores radians
//! and the raw pose fields.
//!
//! Readers reject unknown major versions and ignore unknown minor fields; the
//! structs deliberately do not use `deny_unknown_fields`.

use serde::{Deserialize, Serialize};

use crate::score::{Detection, DetectionDiagnostics};
use crate::search::{DetectionResult, SearchParams, SearchTimings};

/// Result-file format identifier.
pub const RESULTS_FORMAT: &str = "pattern-match-results";
/// Current result-file schema version.
pub const RESULTS_VERSION: u32 = 1;

/// Test-image dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageInfo {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

/// Reported pose at the model zero (README §4, §12.3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseRecord {
    /// Image x of the model zero.
    pub x: f64,
    /// Image y of the model zero.
    pub y: f64,
    /// Local angle in degrees, in `(-180, 180]`.
    pub angle_deg: f64,
    /// Local scale relative to the taught size.
    pub scale: f64,
}

/// Stage timings in milliseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TimingsRecord {
    /// Blurred base image and pyramid construction.
    pub preprocess: f64,
    /// Gradients, edges, distance fields, corners, and segments.
    pub features: f64,
    /// Coarse hypothesis generation.
    pub search: f64,
    /// Similarity refinement, homography estimation, and NMS.
    pub refine: f64,
    /// End-to-end time.
    pub total: f64,
}

impl From<&SearchTimings> for TimingsRecord {
    fn from(timings: &SearchTimings) -> Self {
        Self {
            preprocess: timings.preprocess_ms,
            features: timings.features_ms,
            search: timings.search_ms,
            refine: timings.refine_ms,
            total: timings.total_ms,
        }
    }
}

/// One serialized detection (README §12.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectionRecord {
    /// Sequential detection id.
    pub id: usize,
    /// Reported pose at the model zero.
    pub pose: PoseRecord,
    /// 3×3 row-major homography mapping model to image.
    pub homography: [[f64; 3]; 3],
    /// Whether the homography is perspective evidence rather than a similarity
    /// embedding.
    pub homography_estimated: bool,
    /// Weighted edge score.
    pub score: f32,
    /// Heuristic confidence.
    pub confidence: f32,
    /// Auditable diagnostics.
    pub diagnostics: DetectionDiagnostics,
}

impl From<&Detection> for DetectionRecord {
    fn from(detection: &Detection) -> Self {
        Self {
            id: detection.id,
            pose: PoseRecord {
                x: detection.pose.tx,
                y: detection.pose.ty,
                angle_deg: normalize_degrees(detection.pose.theta),
                scale: detection.pose.scale,
            },
            homography: detection.homography,
            homography_estimated: detection.homography_estimated,
            score: detection.score,
            confidence: detection.confidence,
            diagnostics: detection.diagnostics.clone(),
        }
    }
}

/// A complete versioned result document (README §12.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultsFile {
    /// Always [`RESULTS_FORMAT`].
    pub format: String,
    /// Major schema version.
    pub version: u32,
    /// Model name.
    pub model: String,
    /// Test-image dimensions.
    pub image: ImageInfo,
    /// Search parameters used for the run.
    pub parameters: SearchParams,
    /// Stage timings.
    pub timings_ms: TimingsRecord,
    /// Detections after thresholding and NMS.
    pub detections: Vec<DetectionRecord>,
}

impl ResultsFile {
    /// Build a result document from a detection run.
    #[must_use]
    pub fn new(
        model_name: impl Into<String>,
        image_width: usize,
        image_height: usize,
        parameters: &SearchParams,
        result: &DetectionResult,
    ) -> Self {
        Self {
            format: RESULTS_FORMAT.to_owned(),
            version: RESULTS_VERSION,
            model: model_name.into(),
            image: ImageInfo {
                width: image_width,
                height: image_height,
            },
            parameters: *parameters,
            timings_ms: TimingsRecord::from(&result.timings),
            detections: result
                .detections
                .iter()
                .map(DetectionRecord::from)
                .collect(),
        }
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> crate::error::Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| crate::error::Error::Format(error.to_string()))
    }

    /// Parse and validate a result document.  Unknown major versions are
    /// rejected (README §12.3).
    pub fn from_json(json: &str) -> crate::error::Result<Self> {
        let document: Self = serde_json::from_str(json)
            .map_err(|error| crate::error::Error::Format(error.to_string()))?;
        if document.format != RESULTS_FORMAT {
            return Err(crate::error::Error::Format(format!(
                "unexpected result format {:?}",
                document.format
            )));
        }
        if document.version != RESULTS_VERSION {
            return Err(crate::error::Error::Format(format!(
                "unsupported result version {}",
                document.version
            )));
        }
        Ok(document)
    }
}

fn normalize_degrees(theta: f64) -> f64 {
    let mut degrees = theta.to_degrees().rem_euclid(360.0);
    if degrees > 180.0 {
        degrees -= 360.0;
    }
    if degrees <= -180.0 {
        degrees += 360.0;
    }
    degrees
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Similarity;
    use crate::geometry::matrix_to_array;
    use crate::score::{DetectionDiagnostics, SeedSource};

    fn sample_result(homography_estimated: bool) -> DetectionResult {
        let pose = Similarity::from_pose(12.5, -3.25, 0.4, 1.15);
        let homography = if homography_estimated {
            [[1.0, 0.0, 12.5], [0.0, 1.0, -3.25], [0.0002, -0.0001, 1.0]]
        } else {
            matrix_to_array(&pose.embed().normalized_for_output())
        };
        DetectionResult {
            detections: vec![Detection {
                id: 0,
                pose,
                homography,
                homography_estimated,
                score: 0.87,
                confidence: 0.81,
                diagnostics: DetectionDiagnostics {
                    coverage: 0.94,
                    matched_fraction: 0.91,
                    mean_distance_px: 0.62,
                    mean_orientation_agreement: 0.95,
                    seed_source: SeedSource::Hough,
                    correspondences: 184,
                    inliers: 151,
                    ransac_iterations: 87,
                    corner_correspondences: 0,
                    corner_inliers: 0,
                    homography_condition: homography_estimated.then_some(12_500.0),
                    ambiguity: 0.44,
                    fallback_reason: (!homography_estimated)
                        .then(|| "score_not_improved".to_owned()),
                },
            }],
            timings: SearchTimings {
                preprocess_ms: 12.4,
                features_ms: 18.1,
                search_ms: 90.2,
                refine_ms: 22.0,
                total_ms: 124.6,
            },
            hypotheses_generated: 120,
            hypotheses_scored: 24,
        }
    }

    #[test]
    fn result_schema_matches_readme_12_3() {
        let params = SearchParams::default();
        let result = sample_result(true);
        let document = ResultsFile::new("bracket-a", 1280, 1024, &params, &result);
        let json = serde_json::to_value(&document).expect("serialize");
        assert_eq!(json["format"], RESULTS_FORMAT);
        assert_eq!(json["version"], RESULTS_VERSION);
        assert_eq!(json["model"], "bracket-a");
        assert_eq!(json["image"]["width"], 1280);
        assert_eq!(json["image"]["height"], 1024);
        assert!(json["parameters"]["rotation_bins"].is_number());
        assert_eq!(json["timings_ms"]["preprocess"], 12.4);
        assert_eq!(json["timings_ms"]["features"], 18.1);
        assert_eq!(json["timings_ms"]["search"], 90.2);
        assert_eq!(json["timings_ms"]["refine"], 22.0);

        let detection = &json["detections"][0];
        assert_eq!(detection["id"], 0);
        assert!(detection["pose"]["x"].is_number());
        assert!(detection["pose"]["y"].is_number());
        assert!(detection["pose"]["angle_deg"].is_number());
        assert!(detection["pose"]["scale"].is_number());
        assert!(detection["pose"].get("theta").is_none());
        assert!(detection["homography"].is_array());
        assert_eq!(detection["homography"].as_array().unwrap().len(), 3);
        assert_eq!(detection["homography_estimated"], true);
        assert_eq!(detection["diagnostics"]["seed_source"], "hough");
        assert_eq!(detection["diagnostics"]["correspondences"], 184);
        assert_eq!(detection["diagnostics"]["inliers"], 151);
        assert_eq!(detection["diagnostics"]["ransac_iterations"], 87);
        assert_eq!(detection["diagnostics"]["homography_condition"], 12_500.0);
        assert!(detection["diagnostics"]["fallback_reason"].is_null());
        assert!(
            detection["diagnostics"].get("corner_inliers").is_none(),
            "internal corner diagnostics must stay out of the result schema"
        );
    }

    #[test]
    fn result_json_roundtrip_preserves_detections() {
        let params = SearchParams::default();
        let result = sample_result(false);
        let document = ResultsFile::new("bracket-a", 640, 480, &params, &result);
        let json = document.to_json().expect("serialize");
        let parsed = ResultsFile::from_json(&json).expect("parse");
        assert_eq!(document, parsed);
        assert_eq!(parsed.detections[0].pose.angle_deg, {
            let mut angle = (0.4f64).to_degrees().rem_euclid(360.0);
            if angle > 180.0 {
                angle -= 360.0;
            }
            angle
        });
        assert_eq!(
            parsed.detections[0].diagnostics.fallback_reason.as_deref(),
            Some("score_not_improved")
        );
    }

    #[test]
    fn result_reader_rejects_unknown_version() {
        let params = SearchParams::default();
        let result = sample_result(true);
        let mut document = ResultsFile::new("bracket-a", 64, 64, &params, &result);
        document.version = RESULTS_VERSION + 1;
        let json = serde_json::to_string(&document).expect("serialize");
        assert!(ResultsFile::from_json(&json).is_err());

        document.version = RESULTS_VERSION;
        document.format = "something-else".to_owned();
        let json = serde_json::to_string(&document).expect("serialize");
        assert!(ResultsFile::from_json(&json).is_err());
    }

    #[test]
    fn pose_angle_is_reported_in_documented_range() {
        for (radians, expected) in [
            (0.0, 0.0),
            (std::f64::consts::PI, 180.0),
            (-std::f64::consts::PI, 180.0),
            (3.0 * std::f64::consts::PI / 2.0, -90.0),
            (-3.0 * std::f64::consts::PI / 2.0, 90.0),
        ] {
            let actual = normalize_degrees(radians);
            assert!(
                (actual - expected).abs() < 1e-9,
                "angle {radians} -> {actual}, expected {expected}"
            );
        }
    }
}
