//! `pm-cli bench synthetic`: deterministic synthetic benchmark.

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use pm_core::{BenchmarkReport, ScenarioReport, SyntheticBenchParams, run_benchmark_with_params};

use crate::common;

/// Arguments for `pm-cli bench synthetic`.
#[derive(Debug, Args)]
pub struct SyntheticArgs {
    /// Path to the `.pmodel` model file (must embed its source image).
    #[arg(long, value_name = "MODEL")]
    pub model: PathBuf,
    /// Number of clean-similarity cases; other scenarios use a fraction.
    #[arg(long, default_value_t = 200)]
    pub cases: usize,
    /// Deterministic RNG seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Optional benchmark report JSON output path.
    #[arg(long, value_name = "FILE")]
    pub report: Option<PathBuf>,
    /// Partial or full parameter override JSON.
    #[arg(long, value_name = "FILE")]
    pub params: Option<PathBuf>,
    /// Accuracy mode options.
    #[command(flatten)]
    pub accuracy: common::AccuracyMode,
}

/// Execute `pm-cli bench synthetic`.
pub fn execute(args: SyntheticArgs) -> Result<()> {
    let model = common::load_model(&args.model)?;
    let params = common::load_params(&model, args.params.as_deref(), args.accuracy.high_accuracy)?;
    let bench_params = SyntheticBenchParams {
        cases: args.cases,
        seed: args.seed,
    };
    let report = run_benchmark_with_params(&model, &bench_params, &params)?;
    print_report(&report);

    if let Some(path) = &args.report {
        common::write_text(path, &report.to_json()?)?;
        println!("report written to {}", path.display());
    }
    Ok(())
}

fn print_report(report: &BenchmarkReport) {
    println!(
        "synthetic benchmark: model {:?}, seed {}, {} clean case(s)",
        report.model, report.seed, report.cases
    );
    println!(
        "{:<30} {:>6} {:>7} {:>8} {:>10} {:>9} {:>10} {:>7} {:>9} {:>10}",
        "scenario",
        "cases",
        "recall",
        "FP/img",
        "max pos px",
        "max ang°",
        "max scale%",
        "H acc",
        "fallback",
        "max reproj"
    );
    for scenario in &report.scenarios {
        print_scenario(scenario);
    }
}

fn print_scenario(scenario: &ScenarioReport) {
    let reprojection = scenario
        .max_roi_reprojection_px
        .map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"));
    let recall = if scenario.ground_truth == 0 {
        "-".to_owned()
    } else {
        format!("{:.3}", scenario.recall)
    };
    let accepted = if scenario.true_positives == 0 {
        "-".to_owned()
    } else {
        format!("{:.3}", scenario.homography_accepted_fraction)
    };
    println!(
        "{:<30} {:>6} {:>7} {:>8.3} {:>10.3} {:>9.3} {:>10.3} {:>7} {:>9} {:>10}",
        scenario.name,
        scenario.cases,
        recall,
        scenario.false_positives_per_image,
        scenario.max_position_error_px,
        scenario.max_angle_error_deg,
        scenario.max_scale_error_percent,
        accepted,
        scenario.homography_fallback,
        reprojection
    );
}
