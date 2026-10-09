//! arbvis: visualize arbitrary binary files laid out along a Hilbert curve.
//!
//! This crate is the byte-only foundation
//! [modelweightvis](https://github.com/znation/modelweightvis)
//! builds on: tile pipeline, source/diff plumbing, layout traits, and hooks
//! for the model-aware plugins to plug into.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

mod cli;
pub mod color;
pub mod data;
pub mod data_diff;
mod deploy;
mod geometry;
pub mod hf_cli;
mod hf_upload;
pub mod hf_url;
mod json_diff;
mod layout;
mod perf_monitor;
mod pipeline;
mod progress;
mod providers;
pub mod registry;
mod throttle;
mod tiled;
mod volume;
pub mod xet;

// Public library surface — the byte-only foundation modelweightvis builds
// on. Tile pipeline, source/diff plumbing, layout traits, hooks for the
// model-aware plugins to plug into.
pub use cli::Args;
pub use data::{
    load_source_data, prepare_sources, prepare_sources_from_specs, CustomSource, Data, DiffFill,
    Extensions, LazyFetcher, SceneTag, Source, SourceKind,
};
pub use data_diff::{byte_directory_diff, prepare_diff_sources};
pub use geometry::name_hue;
pub use layout::{CanvasGeom, LayoutMode, LayoutShape};
pub use pipeline::run;
pub use registry::{
    Branding, DestKind, DiffBuildCtx, DiffPair, DiffSourceBuilder, FormatPlugin, LayoutBuildCtx,
    LayoutPlugin, PrepareSourcesExtension, Registry, RenderHints, SourceCtx, SourceProvider,
    VolumeShapePlugin,
};
pub use tiled::html::FileEntity;
pub use tiled::leaf::{encode_tile, TileFormat, TILE};
pub use tiled::leaf_renderer::{
    LeafLoader, LeafRegistry, LeafRenderer, LeafTile, LoadCtx, RenderCtx,
};
pub use tiled::{EncodedTile, LeafMode, LoadedTile};
// 3D (`--3d`) volume seam — the placement + coloring analogs of the 2D layout
// and leaf-renderer SPI, for a downstream to render a structure-aware cube.
pub use volume::{
    VolumeEntity, VolumeLabel, VolumeShape, VoxelBox, VoxelCell, VoxelGridMut, VoxelRegistry,
    VoxelRenderCtx, VoxelRenderer,
};

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// One-time process-global init: env vars + logger + rayon pool + tokio
/// runtime. Returns the runtime so the caller can `block_on(run(...))`.
///
/// Must be called before any other thread is spawned (the `set_var` calls
/// are not thread-safe and the rayon global pool can only be initialised
/// once). The binary entrypoint is the only intended caller.
pub fn init() -> anyhow::Result<tokio::runtime::Runtime> {
    // anyhow only captures backtraces when this env var is set. Default it
    // on so any error that bubbles up shows where it originated — we'd
    // rather pay the per-`anyhow!()` backtrace cost than debug blind.
    // SAFETY: this runs on the single main thread before any other thread
    // could touch the environment. set_var has been marked unsafe on recent
    // Rust toolchains.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        // SAFETY: see comment above.
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "1");
        }
    }
    // Belt-and-suspenders for the rav1e stack appetite (see runtime build
    // below). RUST_MIN_STACK is read by `std::thread::Builder` whenever a
    // builder doesn't explicitly set `stack_size` — so any third-party crate
    // (xet-runtime, hf-xet, …) that spawns its own threads with default
    // settings inherits this floor. Has to be set before *any* thread is
    // spawned. SAFETY: same as above — single main thread, pre-spawn.
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        // SAFETY: see comment above.
        unsafe {
            std::env::set_var("RUST_MIN_STACK", (8 * 1024 * 1024).to_string());
        }
    }

    // Build env_logger but DON'T install it directly; wrap it in `LogWrapper`
    // so every log line is printed via `MultiProgress::suspend(...)`, which
    // pauses bar rendering, writes the line cleanly, and redraws the bars.
    // Without this, concurrent progress bars and log output overwrite each
    // other and the bar stack drifts down the screen line-by-line.
    let env_logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    let max_level = env_logger.filter();
    indicatif_log_bridge::LogWrapper::new(progress::multi().clone(), env_logger)
        .try_init()
        .expect("global logger already set");
    log::set_max_level(max_level);

    // The AVIF encoder underneath `image::codecs::avif::AvifEncoder` is rav1e,
    // which uses `rayon` internally to parallelize AV1 tile encoding. Per-call
    // rav1e needs ≥4 MB of stack; Rust's default `std::thread` stack is 2 MB
    // on macOS. We have to bump the stack on EVERY thread that might run rav1e:
    //   - tokio worker pool + blocking pool (we spawn the AVIF encode via
    //     `tokio::task::spawn_blocking` from the pyramid accumulator), and
    //   - rayon's global thread pool (rav1e's internal parallelism).
    //
    // The previous fix only handled tokio; the unnamed thread that overflowed
    // mid-pyramid was a rayon worker spun up the first time rav1e tried to
    // parallelize. Initialise rayon's global pool with the larger stack BEFORE
    // any other code touches it (a previous accidental rayon call would lock
    // in the default-stack pool for the process lifetime).
    rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global()
        .map_err(|e| anyhow::anyhow!("building rayon global pool: {e}"))?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(8 * 1024 * 1024)
        .build()
        .map_err(|e| anyhow::anyhow!("building tokio runtime: {e}"))?;
    Ok(rt)
}

/// Optional perf monitor (set `ARBVIS_PERF_LOG=1`) — emits one line/s with
/// throttle + CAS HTTP counters. Used to localise pipeline stalls. The
/// returned `Arc<AtomicBool>` is the run-flag; drop it (or set it to false)
/// to stop the monitor. Bind to a named local in `main` to keep it alive for
/// the process lifetime.
pub fn perf_monitor_spawn_if_enabled() -> Option<Arc<AtomicBool>> {
    perf_monitor::spawn_if_enabled()
}
