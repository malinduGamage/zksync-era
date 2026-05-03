use std::path::PathBuf;

use anyhow::Context as _;
use structopt::StructOpt;

use batch_calibration::runner::run;

#[derive(Debug, StructOpt)]
#[structopt(name = "Batch calibration runner", author = "Matter Labs")]
struct Opt {
    #[structopt(long)]
    core_db_url: String,

    #[structopt(long)]
    prover_db_url: String,

    #[structopt(long)]
    telemetry_jsonl: Option<PathBuf>,

    #[structopt(long)]
    output_dir: PathBuf,

    #[structopt(long)]
    start_batch: Option<u32>,

    #[structopt(long)]
    end_batch: Option<u32>,

    #[structopt(long)]
    protocol_version: Option<String>,

    #[structopt(long, default_value = "20")]
    minimum_samples: usize,

    #[structopt(long, default_value = "5")]
    folds: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    run(
        &opt.core_db_url,
        &opt.prover_db_url,
        opt.telemetry_jsonl.as_deref(),
        &opt.output_dir,
        opt.start_batch,
        opt.end_batch,
        opt.protocol_version.as_deref(),
        opt.minimum_samples,
        opt.folds,
    )
    .await
    .context("batch calibration failed")
}
