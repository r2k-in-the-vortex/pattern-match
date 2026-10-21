//! `pm-cli` — headless Pattern Match command-line interface.
//!
//! The command surface follows README §12.2: `run`, `batch`,
//! `model inspect`, and `bench synthetic`.  Every command loads the same
//! `pm-core` model and runs the same deterministic detector as the library and
//! UI, so results are reproducible from the stored parameters and seeds.

mod batch;
mod bench;
mod common;
mod inspect;
mod run;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "pm-cli", version, about = "Headless pattern-match detection")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run detection on a single image.
    Run(run::RunArgs),
    /// Run detection over a directory of images.
    Batch(batch::BatchArgs),
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

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Print model metadata and feature counts.
    Inspect(inspect::InspectArgs),
}

#[derive(Debug, Subcommand)]
enum BenchCommand {
    /// Run the deterministic synthetic benchmark.
    Synthetic(bench::SyntheticArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run::execute(args),
        Command::Batch(args) => batch::execute(args),
        Command::Model {
            command: ModelCommand::Inspect(args),
        } => inspect::execute(args),
        Command::Bench {
            command: BenchCommand::Synthetic(args),
        } => bench::execute(args),
    }
}
