//! Shared `indicatif` style helpers.
//!
//! Three styles cover the patterns used across the crate:
//! - [`counter_style`]: a left-anchored bar tracking total work done, with rate
//!   and ETA suffix (`{per_sec}, ETA {eta}`).
//! - [`queue_style`]: an indented secondary bar for instantaneous queue fill,
//!   anchored under whatever counter it follows.
//! - [`status_style`]: no bar — just a status message and the elapsed time.
//!   Used for the AIMD throttle indicator.

use std::sync::OnceLock;

use indicatif::{MultiProgress, ProgressDrawTarget, ProgressStyle};

/// Process-wide `MultiProgress` that owns every progress bar across the run.
///
/// All bars (whether grouped under a `PipelineProgress`, standalone counters,
/// or upload trackers) attach themselves to this `MultiProgress` so they
/// share a single draw target and renderer. Log output is routed through it
/// via `indicatif-log-bridge` (see `LogWrapper` in `main`), so any `log::*`
/// call automatically suspends bar rendering, prints, and resumes — instead
/// of overwriting bar lines.
///
/// The draw target is stderr; if stderr isn't a TTY, indicatif's hidden
/// target is used so non-interactive runs stay log-only.
pub fn multi() -> &'static MultiProgress {
    static M: OnceLock<MultiProgress> = OnceLock::new();
    M.get_or_init(|| {
        use std::io::IsTerminal;
        if std::io::stderr().is_terminal() {
            MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
        } else {
            MultiProgress::with_draw_target(ProgressDrawTarget::hidden())
        }
    })
}

pub fn counter_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg} ({per_sec}, ETA {eta})",
    )
    .expect("counter_style template parse")
    .progress_chars("=>-")
}

pub fn queue_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  └─ {msg}: {pos}/{len} [{bar:30.yellow/dim}]",
    )
    .expect("queue_style template parse")
    .progress_chars("=>-")
}

pub fn status_style() -> ProgressStyle {
    ProgressStyle::with_template("{msg}  ({elapsed_precise})")
        .expect("status_style template parse")
}
