//! Shared CLI helpers: model/parameter loading and output writing.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::Args;
use pm_core::{
    Detection, DetectionOverlayOptions, Model, RgbaImage, SearchParams, SearchTimings,
    model_roi_in_model_coords, render_detection_overlay,
};

/// Shared accuracy-mode flag for commands that build search parameters.
#[derive(Debug, Args)]
pub struct AccuracyMode {
    /// Use the high-accuracy Hough orientation gate (slower Hough, slightly
    /// better recall on low-contrast and cluttered scenes).
    #[arg(long)]
    pub high_accuracy: bool,
}

/// Load a `.pmodel` file with a path-qualified error.
pub fn load_model(path: &Path) -> Result<Model> {
    Model::load(path).with_context(|| format!("load model {}", path.display()))
}

/// Build the search parameters from the model defaults, optionally overridden by
/// a JSON file.
///
/// The override may be a complete [`SearchParams`] document, a partial object
/// (missing fields keep the model defaults, nested objects merge), or a result
/// file containing a `parameters` object.  Unknown keys are rejected so a typo
/// cannot silently change nothing.  `high_accuracy` overrides the orientation
/// gate after the file is applied.
pub fn load_params(
    model: &Model,
    path: Option<&Path>,
    high_accuracy: bool,
) -> Result<SearchParams> {
    let base = SearchParams::from(model.search_defaults);
    let mut params = if let Some(path) = path {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read parameter file {}", path.display()))?;
        let mut user: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parse parameter file {}", path.display()))?;
        if let Some(parameters) = user.get("parameters") {
            user = parameters.clone();
        }
        if !user.is_object() {
            bail!(
                "parameter file {} must contain a JSON object",
                path.display()
            );
        }
        let mut merged = serde_json::to_value(base).expect("SearchParams serializes");
        reject_unknown_keys(&merged, &user, path)?;
        merge_values(&mut merged, &user);
        serde_json::from_value(merged)
            .with_context(|| format!("parameter file {} has invalid values", path.display()))?
    } else {
        base
    };
    if high_accuracy {
        params.orientation_gate = SearchParams::HIGH_ACCURACY_ORIENTATION_GATE;
    }
    Ok(params)
}

fn reject_unknown_keys(
    base: &serde_json::Value,
    user: &serde_json::Value,
    path: &Path,
) -> Result<()> {
    let (Some(base_object), Some(user_object)) = (base.as_object(), user.as_object()) else {
        return Ok(());
    };
    for (key, value) in user_object {
        let Some(base_value) = base_object.get(key) else {
            bail!(
                "parameter file {}: unknown parameter {key:?}",
                path.display()
            );
        };
        reject_unknown_keys(base_value, value, path)?;
    }
    Ok(())
}

fn merge_values(base: &mut serde_json::Value, user: &serde_json::Value) {
    match (base, user) {
        (serde_json::Value::Object(base_object), serde_json::Value::Object(user_object)) => {
            for (key, value) in user_object {
                if let Some(base_value) = base_object.get_mut(key) {
                    merge_values(base_value, value);
                } else {
                    base_object.insert(key.clone(), value.clone());
                }
            }
        }
        (base, user) => *base = user.clone(),
    }
}

/// Create the parent directory if needed and write `bytes`.
pub fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// Write a string as UTF-8.
pub fn write_text(path: &Path, text: &str) -> Result<()> {
    write_file(path, text.as_bytes())
}

/// Print one line of stage timings.
pub fn print_timings(timings: &SearchTimings) {
    println!(
        "timings: preprocess {:.1} ms, features {:.1} ms, search {:.1} ms, \
         refine {:.1} ms, total {:.1} ms",
        timings.preprocess_ms,
        timings.features_ms,
        timings.search_ms,
        timings.refine_ms,
        timings.total_ms
    );
}

/// Print one summary line per detection.
pub fn print_detections(detections: &[Detection]) {
    for detection in detections {
        let homography = if detection.homography_estimated {
            "estimated".to_owned()
        } else {
            match &detection.diagnostics.fallback_reason {
                Some(reason) => format!("fallback: {reason}"),
                None => "fallback".to_owned(),
            }
        };
        println!(
            "  [{}] x={:.2} y={:.2} angle={:.2}° scale={:.3} score={:.3} conf={:.3} \
             seed={} H={}",
            detection.id,
            detection.pose.tx,
            detection.pose.ty,
            detection.pose.theta.to_degrees(),
            detection.pose.scale,
            detection.score,
            detection.confidence,
            detection.diagnostics.seed_source.as_str(),
            homography
        );
    }
}

/// Render detection geometry over the display image and save it as a PNG.
pub fn write_overlay(
    path: &Path,
    display: &RgbaImage,
    model: &Model,
    detections: &[Detection],
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    let roi_model = model_roi_in_model_coords(model);
    let overlay = render_detection_overlay(
        display,
        &roi_model,
        &model.features.edges.points,
        detections,
        &DetectionOverlayOptions::default(),
    );
    overlay
        .save(path)
        .with_context(|| format!("write overlay {}", path.display()))
}
