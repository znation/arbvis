//! Thin binary entrypoint for `modelweightvis`.
//!
//! Reuses arbvis's CLI surface, runtime init, and dispatch — diverges from
//! the byte-only `arbvis` binary by registering tensor-aware plugins
//! (arch / MoE-diff layouts, tensor-diff source builder, arch leaf
//! loader+renderer) on top of the default registry.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let rt = arbvis::init()?;
    // Bind to a named local so the run-flag stays alive for the process
    // lifetime (drop stops the monitor). See `perf_monitor::spawn_if_enabled`.
    let _perf_monitor_stop = arbvis::perf_monitor_spawn_if_enabled();
    let args = arbvis::Args::parse();
    let mut registry = arbvis::Registry::with_defaults();
    modelweightvis::register_all(&mut registry);
    rt.block_on(arbvis::run(args, registry))
}
