//! Thin binary entrypoint for `modelweightvis`.
//!
//! Diverges from the byte-only `arbvis` binary on two axes:
//! - Uses `modelweightvis::Args` (clap-flatten of `arbvis::Args` + the
//!   four tensor-aware flags `--moe-diff`, `--finetune` / `--no-finetune`,
//!   `--diff-metric`, `--layout`) instead of `arbvis::Args` directly.
//! - Registers tensor-aware plugins (format parsers, arch + MoE-diff
//!   layouts, tensor-diff source builder, arch leaf loader+renderer,
//!   `SourceMetaSidecarHook`, and the option-slot hooks) on top of the
//!   default registry via `register_all`.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let rt = arbvis::init()?;
    // Bind to a named local so the run-flag stays alive for the process
    // lifetime (drop stops the monitor). See `perf_monitor::spawn_if_enabled`.
    let _perf_monitor_stop = arbvis::perf_monitor_spawn_if_enabled();
    let (args, opts) = modelweightvis::Args::parse().split();
    let mut registry = arbvis::Registry::with_defaults();
    modelweightvis::register_all(&mut registry);
    rt.block_on(arbvis::run(args, opts, registry))
}
