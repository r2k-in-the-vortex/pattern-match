//! `pm-cli run`: single-image detection with JSON, CSV, and overlay output.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Args;
use pm_core::{ResultsFile, detect, detections_to_csv, load_image};

use crate::common;

/// Arguments for `pm-cli run`.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Path to the `.pmodel` model file.
    #[arg(long, value_name = "MODEL")]
    pub model: PathBuf,
    /// Test image to search.
    #[arg(long, value_name = "IMAGE")]
    pub image: PathBuf,
    /// Result JSON output path.
    #[arg(long, value_name = "FILE")]
    pub output: PathBuf,
    /// Optional detection CSV output path.
    #[arg(long, value_name = "FILE")]
    pub csv: Option<PathBuf>,
    /// Optional annotated overlay PNG output path.
    #[arg(long, value_name = "FILE")]
    pub overlay: Option<PathBuf>,
    /// Partial or full parameter override JSON.
    #[arg(long, value_name = "FILE")]
    pub params: Option<PathBuf>,
    /// Print stage timings.
    #[arg(long)]
    pub timings: bool,
    /// Exit non-zero when no detections are found.
    #[arg(long)]
    pub fail_on_none: bool,
}

/// Execute `pm-cli run`.
pub fn execute(args: RunArgs) -> Result<()> {
    let model = common::load_model(&args.model)?;
    let params = common::load_params(&model, args.params.as_deref())?;
    let image =
        load_image(&args.image).with_context(|| format!("load image {}", args.image.display()))?;
    let result = detect(&model, &image.gray, &params)
        .with_context(|| format!("detect in {}", args.image.display()))?;
    let document = ResultsFile::new(
        &model.name,
        image.gray.width,
        image.gray.height,
        &params,
        &result,
    );

    common::write_text(&args.output, &document.to_json()?)?;
    if let Some(csv_path) = &args.csv {
        common::write_text(csv_path, &detections_to_csv(&document.detections))?;
    }
    if let Some(overlay_path) = &args.overlay {
        common::write_overlay(overlay_path, &image.display, &model, &result.detections)?;
    }

    if args.timings {
        common::print_timings(&result.timings);
    }
    println!(
        "{}: {} detection(s) written to {}",
        args.image.display(),
        result.detections.len(),
        args.output.display()
    );
    common::print_detections(&result.detections);

    if args.fail_on_none && result.detections.is_empty() {
        bail!("no detections found in {}", args.image.display());
    }
    Ok(())
}
