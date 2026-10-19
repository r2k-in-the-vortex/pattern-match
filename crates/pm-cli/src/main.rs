//! `pm-cli` — headless Pattern Match command-line interface.
//!
//! The command surface follows README §12.2. Parsing is wired up in M0; the
//! detection pipeline arrives in M5–M7.

use std::path::PathBuf;

use anyhow::bail;
use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "pm-cli", version, about = "Headless pattern-match detection")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run detection on a single image.
    Run(RunArgs),
    /// Run detection over a directory of images.
    Batch(BatchArgs),
    /// Inspect model files.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Run benchmarks.
    Bench {
        #[command(subcommand)]
        command: BenchCommand,
    },
}

/// Arguments for `pm-cli run`.
#[allow(dead_code, reason = "consumed by the M7 implementation")]
#[derive(Debug, Args)]
struct RunArgs {
    /// Path to the `.pmodel` model file.
    #[arg(long, value_name = "MODEL")]
    model: PathBuf,
    /// Test image to search.
    #[arg(long, value_name = "IMAGE")]
    image: PathBuf,
    /// Result JSON output path.
    #[arg(long, value_name = "FILE")]
    output: PathBuf,
    /// Optional annotated overlay PNG output path.
    #[arg(long, value_name = "FILE")]
    overlay: Option<PathBuf>,
    /// Full parameter override JSON.
    #[arg(long, value_name = "FILE")]
    params: Option<PathBuf>,
    /// Print stage timings.
    #[arg(long)]
    timings: bool,
    /// Exit non-zero when no detections are found.
    #[arg(long)]
    fail_on_none: bool,
}

/// Arguments for `pm-cli batch`.
#[allow(dead_code, reason = "consumed by the M7 implementation")]
#[derive(Debug, Args)]
struct BatchArgs {
    /// Path to the `.pmodel` model file.
    #[arg(long, value_name = "MODEL")]
    model: PathBuf,
    /// Directory of test images.
    #[arg(long, value_name = "DIR")]
    input: PathBuf,
    /// Directory receiving result files.
    #[arg(long, value_name = "DIR")]
    output_dir: PathBuf,
    /// Recurse into subdirectories.
    #[arg(long)]
    recursive: bool,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Print model metadata and feature counts.
    Inspect {
        /// Path to the `.pmodel` model file.
        #[arg(long, value_name = "MODEL")]
        model: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    /// Run the deterministic synthetic benchmark.
    Synthetic {
        /// Path to the `.pmodel` model file.
        #[arg(long, value_name = "MODEL")]
        model: PathBuf,
        /// Number of benchmark cases.
        #[arg(long, default_value_t = 200)]
        cases: u32,
        /// Deterministic RNG seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run(_) => bail!("pm-cli run is not implemented yet (arrives in M5–M7)"),
        Command::Batch(_) => bail!("pm-cli batch is not implemented yet (arrives in M7)"),
        Command::Model { .. } => bail!("pm-cli model is not implemented yet (arrives in M3/M7)"),
        Command::Bench { .. } => bail!("pm-cli bench is not implemented yet (arrives in M7)"),
    }
}
