//! Thin binary entrypoint for `modelweightvis`.
//!
//! Today this is functionally identical to the `arbvis` binary — it
//! constructs the same default registry and hands off to `arbvis::run`. Step
//! 12d moves the tensor-format / arch-layout / tensor-diff plugins out of
//! arbvis into the `modelweightvis` library and registers them here on top
//! of arbvis's defaults so the two binaries diverge.

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
