//! Thin binary entrypoint: hand off to the `arbvis` library.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let rt = arbvis::init()?;
    // Bind to a named local so the run-flag stays alive for the process
    // lifetime (drop stops the monitor). See `perf_monitor::spawn_if_enabled`.
    let _perf_monitor_stop = arbvis::perf_monitor_spawn_if_enabled();
    let args = arbvis::Args::parse();
    let registry = arbvis::Registry::with_defaults();
    rt.block_on(arbvis::run(args, registry))
}
