//! Shared `indicatif` style helpers.
//!
//! Three styles cover the patterns used across the crate:
//! - [`counter_style`]: a left-anchored bar tracking total work done, with rate
//!   and ETA suffix (`{per_sec}, ETA {eta}`).
//! - [`queue_style`]: an indented secondary bar for instantaneous queue fill,
//!   anchored under whatever counter it follows.
//! - [`status_style`]: no bar — just a status message and the elapsed time.
//!   Used for the AIMD throttle indicator.

use indicatif::ProgressStyle;

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
