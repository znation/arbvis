//! Thin binary entrypoint: hand off to the `arbvis` library.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let rt = arbvis::init()?;
    // Bind to a named local so the run-flag stays alive for the process
    // lifetime (drop stops the monitor). See `perf_monitor::spawn_if_enabled`.
    let _perf_monitor_stop = arbvis::perf_monitor_spawn_if_enabled();
    let args = arbvis::Args::parse();
    // The byte-only arbvis CLI registers just the built-in byte providers /
    // Hilbert layout. A downstream specialization extends the registry and maps
    // its own flags onto it before calling `run`.
    let registry = arbvis::Registry::with_defaults();
    rt.block_on(arbvis::run(args, registry))
}
