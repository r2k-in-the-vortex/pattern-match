//! `pm-cli model inspect`: print model metadata and feature counts.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use pm_core::Model;
use pm_core::model::{MODEL_FORMAT, MODEL_VERSION};

use crate::common;

/// Arguments for `pm-cli model inspect`.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Path to the `.pmodel` model file.
    #[arg(long, value_name = "MODEL")]
    pub model: PathBuf,
    /// Print the summary as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Execute `pm-cli model inspect`.
pub fn execute(args: InspectArgs) -> Result<()> {
    let model = common::load_model(&args.model)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary_json(&model))?);
    } else {
        print_summary(&model);
    }
    Ok(())
}

fn print_summary(model: &Model) {
    let (edges, chains, corners, segments, descriptors, pairs) = model.counts();
    println!("model: {}", model.name);
    println!("format: {MODEL_FORMAT} v{MODEL_VERSION}");
    println!("created_unix_seconds: {}", model.created_unix_seconds);
    println!(
        "source: {}x{} ({})",
        model.source_size.0,
        model.source_size.1,
        if model.source.is_some() {
            "embedded"
        } else {
            "not embedded"
        }
    );
    println!(
        "roi: center ({:.2}, {:.2}), {:.2} x {:.2}, angle {:.3} rad",
        model.roi.cx, model.roi.cy, model.roi.width, model.roi.height, model.roi.angle
    );
    println!(
        "zero: origin ({:.2}, {:.2}), angle {:.3} rad",
        model.zero.origin[0], model.zero.origin[1], model.zero.angle
    );
    println!(
        "features: edges {edges}, chains {chains}, corners {corners}, segments {segments}, \
         descriptors {descriptors}, segment_pairs {pairs}"
    );
    println!(
        "mask: {} primitive(s), resolution {:.3}, excluded weight {:.3} ({:.2}%)",
        model.mask_primitives.len(),
        model.mask_resolution,
        model.excluded_edge_weight,
        model.excluded_edge_weight_fraction * 100.0
    );
    let defaults = model.search_defaults;
    println!(
        "search defaults: rotation [{:.1}, {:.1}]°, scale [{:.3}, {:.3}], max detections {}, \
         threshold {:.3}, coverage {:.3}, polarity {}, sources {:?}, RANSAC confidence {:.3}, \
         seed {}",
        defaults.rotation_min_deg,
        defaults.rotation_max_deg,
        defaults.scale_min,
        defaults.scale_max,
        defaults.max_detections,
        defaults.score_threshold,
        defaults.coverage_min,
        defaults.polarity_enabled,
        defaults.hypothesis_sources,
        defaults.ransac_confidence,
        defaults.ransac_seed
    );
    println!(
        "teach params: blur sigma {:.3}, corners {}, segments {}",
        model.params.blur_sigma, model.params.detect_corners, model.params.fit_segments
    );
}

fn summary_json(model: &Model) -> serde_json::Value {
    let (edges, chains, corners, segments, descriptors, pairs) = model.counts();
    serde_json::json!({
        "format": MODEL_FORMAT,
        "version": MODEL_VERSION,
        "name": model.name,
        "created_unix_seconds": model.created_unix_seconds,
        "source": {
            "width": model.source_size.0,
            "height": model.source_size.1,
            "embedded": model.source.is_some(),
        },
        "roi": model.roi,
        "zero": model.zero,
        "features": {
            "edges": edges,
            "chains": chains,
            "corners": corners,
            "segments": segments,
            "descriptors": descriptors,
            "segment_pairs": pairs,
        },
        "mask": {
            "primitives": model.mask_primitives,
            "resolution": model.mask_resolution,
            "bounds": model.mask_bounds,
            "excluded_edge_weight": model.excluded_edge_weight,
            "excluded_edge_weight_fraction": model.excluded_edge_weight_fraction,
        },
        "search_defaults": model.search_defaults,
        "teach_params": model.params,
    })
}
