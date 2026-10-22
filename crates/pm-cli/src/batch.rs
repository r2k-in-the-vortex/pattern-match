//! `pm-cli batch`: detection over a directory of images.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use pm_core::{DetectionRecord, ResultsFile, detect, detections_to_csv_with_images, load_image};

use crate::common;

/// Arguments for `pm-cli batch`.
#[derive(Debug, Args)]
pub struct BatchArgs {
    /// Path to the `.pmodel` model file.
    #[arg(long, value_name = "MODEL")]
    pub model: PathBuf,
    /// Directory of test images.
    #[arg(long, value_name = "DIR")]
    pub input: PathBuf,
    /// Directory receiving result files.
    #[arg(long, value_name = "DIR")]
    pub output_dir: PathBuf,
    /// Recurse into subdirectories.
    #[arg(long)]
    pub recursive: bool,
    /// Optional combined CSV output path.
    #[arg(long, value_name = "FILE")]
    pub csv: Option<PathBuf>,
    /// Optional directory receiving annotated overlay PNGs.
    #[arg(long, value_name = "DIR")]
    pub overlay_dir: Option<PathBuf>,
    /// Partial or full parameter override JSON.
    #[arg(long, value_name = "FILE")]
    pub params: Option<PathBuf>,
    /// Accuracy mode options.
    #[command(flatten)]
    pub accuracy: common::AccuracyMode,
    /// Exit non-zero when any image has no detections.
    #[arg(long)]
    pub fail_on_none: bool,
}

/// Execute `pm-cli batch`.
pub fn execute(args: BatchArgs) -> Result<()> {
    let model = common::load_model(&args.model)?;
    let params = common::load_params(&model, args.params.as_deref(), args.accuracy.high_accuracy)?;
    let images = collect_images(&args.input, args.recursive)?;
    if images.is_empty() {
        bail!("no supported images found in {}", args.input.display());
    }

    let mut used_outputs = HashSet::new();
    let mut csv_rows: Vec<(String, DetectionRecord)> = Vec::new();
    let mut processed = 0usize;
    let mut detections = 0usize;
    let mut failures = 0usize;

    for image_path in &images {
        let relative = image_path.strip_prefix(&args.input).unwrap_or(image_path);
        let display_name = relative.to_string_lossy().replace('\\', "/");
        let output_path = unique_output(&args.output_dir, relative, &mut used_outputs);

        let outcome = process_image(
            &model,
            &params,
            image_path,
            &output_path,
            args.overlay_dir.as_deref(),
            relative,
        );
        match outcome {
            Ok(records) => {
                processed += 1;
                detections += records.len();
                println!("{display_name}: {} detection(s)", records.len());
                if records.is_empty() && args.fail_on_none {
                    eprintln!("{display_name}: no detections");
                    failures += 1;
                }
                csv_rows.extend(
                    records
                        .into_iter()
                        .map(|record| (display_name.clone(), record)),
                );
            }
            Err(error) => {
                failures += 1;
                eprintln!("{display_name}: {error:#}");
            }
        }
    }

    if let Some(csv_path) = &args.csv {
        let rows: Vec<(&str, &DetectionRecord)> = csv_rows
            .iter()
            .map(|(image, record)| (image.as_str(), record))
            .collect();
        common::write_text(csv_path, &detections_to_csv_with_images(&rows))?;
        println!("CSV written to {}", csv_path.display());
    }

    println!(
        "processed {processed}/{} image(s), {detections} detection(s), {failures} failure(s)",
        images.len()
    );
    if failures > 0 {
        bail!("{failures} image(s) failed");
    }
    Ok(())
}

fn process_image(
    model: &pm_core::Model,
    params: &pm_core::SearchParams,
    image_path: &Path,
    output_path: &Path,
    overlay_dir: Option<&Path>,
    relative: &Path,
) -> Result<Vec<DetectionRecord>> {
    let image =
        load_image(image_path).with_context(|| format!("load image {}", image_path.display()))?;
    let result = detect(model, &image.gray, params)
        .with_context(|| format!("detect in {}", image_path.display()))?;
    let document = ResultsFile::new(
        &model.name,
        image.gray.width,
        image.gray.height,
        params,
        &result,
    );
    common::write_text(output_path, &document.to_json()?)?;
    if let Some(directory) = overlay_dir {
        let overlay_path = directory.join(relative).with_extension("overlay.png");
        common::write_overlay(&overlay_path, &image.display, model, &result.detections)?;
    }
    Ok(document.detections)
}

fn unique_output(output_dir: &Path, relative: &Path, used: &mut HashSet<PathBuf>) -> PathBuf {
    let candidate = output_dir.join(relative.with_extension("json"));
    if used.insert(candidate.clone()) {
        return candidate;
    }
    let file_name = relative.file_name().map_or_else(
        || "image".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let candidate = output_dir.join(relative.with_file_name(format!("{file_name}.json")));
    if used.insert(candidate.clone()) {
        return candidate;
    }
    let mut counter = 1usize;
    loop {
        let candidate =
            output_dir.join(relative.with_file_name(format!("{file_name}.{counter}.json")));
        if used.insert(candidate.clone()) {
            return candidate;
        }
        counter += 1;
    }
}

fn collect_images(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let metadata =
        fs::metadata(root).with_context(|| format!("read input directory {}", root.display()))?;
    if !metadata.is_dir() {
        bail!("input {} is not a directory", root.display());
    }

    let mut images = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("read directory {}", directory.display()))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("read directory {}", directory.display()))?;
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    directories.push(path);
                }
            } else if is_supported_image(&path) {
                images.push(path);
            }
        }
    }
    images.sort();
    Ok(images)
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff"
            )
        })
}
