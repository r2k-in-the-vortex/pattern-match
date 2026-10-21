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

/// Detection CSV columns (README §12.3).
pub const DETECTION_CSV_HEADER: &str = "id,seed_source,x,y,angle_deg,scale,h00,h01,h02,h10,h11,h12,h20,h21,h22,homography_estimated,score,confidence,coverage,matched_fraction,mean_distance_px,inliers,correspondences";

/// Serialize detections as CSV rows with the README §12.3 header.
#[must_use]
pub fn detections_to_csv(records: &[DetectionRecord]) -> String {
    let mut output = String::with_capacity(DETECTION_CSV_HEADER.len() + 1 + records.len() * 160);
    output.push_str(DETECTION_CSV_HEADER);
    output.push('\n');
    for record in records {
        output.push_str(&csv_row(record));
        output.push('\n');
    }
    output
}

/// Serialize detections as CSV with a leading `image` column.
///
/// Used by `pm-cli batch`, where rows from several images share one file.
#[must_use]
pub fn detections_to_csv_with_images(rows: &[(&str, &DetectionRecord)]) -> String {
    let mut output = String::with_capacity(6 + DETECTION_CSV_HEADER.len() + 1 + rows.len() * 180);
    output.push_str("image,");
    output.push_str(DETECTION_CSV_HEADER);
    output.push('\n');
    for (image, record) in rows {
        push_csv_field(&mut output, image);
        output.push(',');
        output.push_str(&csv_row(record));
        output.push('\n');
    }
    output
}

fn csv_row(record: &DetectionRecord) -> String {
    let mut output = String::with_capacity(160);
    output.push_str(&record.id.to_string());
    output.push(',');
    output.push_str(record.diagnostics.seed_source.as_str());
    for value in [
        record.pose.x,
        record.pose.y,
        record.pose.angle_deg,
        record.pose.scale,
    ] {
        output.push(',');
        output.push_str(&format_float(value));
    }
    for row in &record.homography {
        for value in row {
            output.push(',');
            output.push_str(&format_float(*value));
        }
    }
    output.push(',');
    output.push_str(if record.homography_estimated {
        "true"
    } else {
        "false"
    });
    for value in [
        f64::from(record.score),
        f64::from(record.confidence),
        f64::from(record.diagnostics.coverage),
        f64::from(record.diagnostics.matched_fraction),
        f64::from(record.diagnostics.mean_distance_px),
    ] {
        output.push(',');
        output.push_str(&format_float(value));
    }
    output.push(',');
    output.push_str(&record.diagnostics.inliers.to_string());
    output.push(',');
    output.push_str(&record.diagnostics.correspondences.to_string());
    output
}

fn format_float(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        format!("{value:.6}")
    }
}

fn push_csv_field(output: &mut String, field: &str) {
    if field.contains([',', '"', '\n', '\r']) {
        output.push('"');
        for character in field.chars() {
            if character == '"' {
                output.push('"');
            }
            output.push(character);
        }
        output.push('"');
    } else {
        output.push_str(field);
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
    fn csv_export_matches_documented_columns() {
        let params = SearchParams::default();
        let result = sample_result(true);
        let document = ResultsFile::new("bracket-a", 1280, 1024, &params, &result);
        let csv = detections_to_csv(&document.detections);
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some(DETECTION_CSV_HEADER));
        let row = lines.next().expect("one detection row");
        assert!(lines.next().is_none());
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(
            fields.len(),
            DETECTION_CSV_HEADER.split(',').count(),
            "column count"
        );
        assert_eq!(fields[0], "0");
        assert_eq!(fields[1], "hough");
        assert_eq!(fields[2], "12.500000");
        assert_eq!(fields[3], "-3.250000");
        assert_eq!(fields[5], "1.150000");
        assert_eq!(fields[15], "true");
        assert_eq!(fields[16], "0.870000");
        assert_eq!(fields[21], "151");
        assert_eq!(fields[22], "184");
    }

    #[test]
    fn csv_with_images_escapes_names() {
        let params = SearchParams::default();
        let result = sample_result(false);
        let document = ResultsFile::new("bracket-a", 640, 480, &params, &result);
        let csv = detections_to_csv_with_images(&[
            ("plain.png", &document.detections[0]),
            ("odd,\"name\".png", &document.detections[0]),
        ]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("image,"));
        assert!(lines[1].starts_with("plain.png,0,"));
        assert!(lines[2].starts_with("\"odd,\"\"name\"\".png\",0,"));
        assert!(lines[1].contains(",false,"));
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
