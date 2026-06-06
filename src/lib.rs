#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod color;
pub mod data;
mod deploy;
mod geometry;
pub mod hf_cli;
mod hf_upload;
pub mod hf_url;
mod json_diff;
mod label;
mod layout;
mod perf_monitor;
mod progress;
mod registry;
mod single;
mod throttle;
mod tiled;
pub mod xet;

// Public library surface — the byte-only foundation modelweightvis builds
// on. Tile pipeline, source/diff plumbing, layout traits, hooks for the
// model-aware plugins to plug into.
pub use data::{
    load_source_data, CustomSource, Data, DiffFill, DiffMetric, Extensions, LazyFetcher, Source,
    SourceKind, SummaryStat,
};
pub use geometry::name_hue;
pub use layout::{CanvasGeom, LayoutMode, LayoutShape};
pub use registry::{
    DiffBuildCtx, DiffSourceBuilder, DirectoryTensorDiffPrep, FinetuneDetect, FormatPlugin,
    LayoutBuildCtx, LayoutPlugin, MoeCkaPrep, MoeDiffPrep, MoeSummaryPrep, PrepareSourcesExtension,
    Registry, RepoDiffPrep, SingleImageArchHook,
};
pub use tiled::html::FileEntity;
pub use tiled::leaf::{encode_tile, TileFormat, TILE};
pub use tiled::leaf_renderer::{
    LeafLoader, LeafRegistry, LeafRenderer, LeafTile, LoadCtx, RenderCtx,
};
pub use tiled::{EncodedTile, LeafMode, LoadedTile};

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use futures::stream::{StreamExt, TryStreamExt};
use tempfile::TempDir;

use crate::data::InputSpec;
use crate::hf_url::HfOutputSpec;

/// Concurrency cap when resolving (downloading) `hf://` inputs at startup.
/// Mirrors `data::SETUP_FETCH_CONCURRENCY` so user-visible parallelism stays
/// consistent across the input-resolution and materialisation stages.
const RESOLVE_CONCURRENCY: usize = 16;

/// Runtime knobs that aren't part of `arbvis::Args` but still flow through
/// `run()`. The byte-only arbvis CLI passes [`ModelOpts::default`] (no MoE
/// diff, no finetune forcing, RMS metric, auto layout — which resolves to
/// Hilbert in a byte-only registry); modelweightvis's CLI shell builds a
/// populated `ModelOpts` from its tensor-aware [`modelweightvis::Args`].
///
/// These knobs all behave correctly with their defaults under arbvis-only
/// registries — `moe_diff = None` is the inert "don't take the MoE branch"
/// case, `layout_mode = Auto` picks the only registered layout (Hilbert),
/// and `diff_metric` only matters when a tensor-aware diff builder is
/// registered. So pure-arbvis callers don't have to reason about these.
#[derive(Clone, Debug)]
pub struct ModelOpts {
    /// `--moe-diff <MODEL>`: render an N×N expert-vs-expert diff matrix for
    /// each MoE layer of a single model. `None` for the byte-only path.
    pub moe_diff: Option<PathBuf>,
    /// `--moe-summary <MODEL>`: render per-expert scalar heatmaps for each
    /// MoE layer (one panel per FFN weight + router). `None` for the
    /// byte-only path. Mutually exclusive with `moe_diff` at the CLI layer.
    pub moe_summary: Option<PathBuf>,
    /// `--moe-cka <MODEL>`: render N×N linear-CKA similarity heatmaps,
    /// one panel per `(layer, weight)`. `None` for the byte-only path.
    /// Mutually exclusive with `moe_diff` and `moe_summary`.
    pub moe_cka: Option<PathBuf>,
    /// `--finetune`: force-treat a `--diff` as a finetune (orig-only
    /// tensors → grey crosshatch, mod-only → error). Exclusive with
    /// `no_finetune`; both `false` means auto-detect via
    /// `FinetuneDetect` hook.
    pub finetune: bool,
    /// `--no-finetune`: force-treat a `--diff` as NOT a finetune.
    pub no_finetune: bool,
    /// `--diff-metric`: how per-element tensor deltas are encoded
    /// (`--diff` / `--moe-diff`). Default is `Rms`.
    pub diff_metric: DiffMetric,
    /// `--summary-stat`: which scalar to compute per expert for
    /// `--moe-summary`. Default is `Rms`.
    pub summary_stat: SummaryStat,
    /// `--cka-sample`: random-projection dimension for `--moe-cka`.
    /// Trades CKA accuracy for compute (smaller = faster, larger =
    /// closer to exact). CLI default is 128.
    pub cka_sample: u32,
    /// `--layout`: structure-aware vs byte-Hilbert layout strategy.
    /// Defaults to `Auto` — `select_layout` iterates the registry's
    /// layout plugins by descending priority and picks the first
    /// applicable. In a byte-only registry that resolves to Hilbert.
    pub layout_mode: LayoutMode,
}

impl Default for ModelOpts {
    fn default() -> Self {
        Self {
            moe_diff: None,
            moe_summary: None,
            moe_cka: None,
            finetune: false,
            no_finetune: false,
            diff_metric: DiffMetric::Rms,
            summary_stat: SummaryStat::Rms,
            cka_sample: 128,
            layout_mode: LayoutMode::Auto,
        }
    }
}

/// CLI tile-format choice. Maps to a `(leaf, pyramid)` pair of [`TileFormat`]s.
///
/// AVIF is the default: ~30-50% smaller than PNG and supported in every
/// modern browser. Pick `png` only for byte-for-byte regression checks or
/// for the rare audience without AVIF support.
#[derive(Clone, Copy, Debug, ValueEnum, Default)]
enum TileFormatArg {
    #[default]
    Avif,
    Png,
}

impl TileFormatArg {
    /// Returns `(leaf_format, pyramid_format)`. Leaf tiles are encoded
    /// near-lossless (each pixel is one source byte; users may inspect),
    /// pyramid tiles are lossy (averaged content tolerates a few QP steps).
    fn split(self) -> (TileFormat, TileFormat) {
        match self {
            TileFormatArg::Avif => (
                TileFormat::Avif {
                    quality: 100,
                    speed: 6,
                },
                TileFormat::Avif {
                    quality: 85,
                    speed: 8,
                },
            ),
            TileFormatArg::Png => (TileFormat::Png, TileFormat::Png),
        }
    }
}

/// Visualize binary files as Hilbert curve plots.
///
/// Each byte is mapped to a color and placed along a Hilbert curve, so
/// structural patterns in the file (e.g. repeated null regions, ASCII text,
/// high-entropy compressed data) become visually apparent.
///
/// Reads from FILES if provided, otherwise reads from stdin.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Files to visualize (defaults to stdin); multiple files are concatenated
    #[arg(conflicts_with = "diff")]
    files: Vec<PathBuf>,

    /// Read file list from this file (one path per line), or - for stdin
    #[arg(short = 'l', long, conflicts_with = "diff")]
    file_list: Option<PathBuf>,

    /// Write the canvas to this PNG file instead of displaying a window.
    /// Accepts a local path or an `hf://` URL (e.g. `hf://datasets/user/repo/path.png`)
    /// to upload directly to the Hub.
    #[arg(short, long, conflicts_with = "tiles")]
    output: Option<PathBuf>,

    /// Write a tiled pyramid for Leaflet.js viewing. Accepts a local directory
    /// (open `index.html` in a browser) or an `hf://` URL to upload the viewer
    /// bundle — `tiles/`, `index.html`, and `labels.json` — to a Hub repo.
    ///
    /// Note: `hf://` upload does NOT stand up a Space; the `index.html` lands in
    /// the target repo but won't render on the Hub on its own. Use `--space` for
    /// a working visualization URL.
    #[arg(short, long, conflicts_with = "output")]
    tiles: Option<PathBuf>,

    /// Visualize abs(modified - original) byte differences; ORIGINAL and MODIFIED are files or directories
    #[arg(long, num_args = 2, value_names = ["ORIGINAL", "MODIFIED"])]
    diff: Option<Vec<PathBuf>>,

    // `--moe-diff`, `--finetune` / `--no-finetune`, `--diff-metric`, and
    // `--layout` previously lived on `Args` too, but they only do anything
    // when a tensor-aware backend is registered. They've moved to
    // `modelweightvis::Args` (which flattens this struct via clap and
    // converts its own fields into [`ModelOpts`]); arbvis-only callers
    // pass [`ModelOpts::default`] to `run()` and never see those flags in
    // `arbvis --help`.
    /// Render tiles and deploy a viewable HF Space (e.g. username/my-vis).
    /// Creates the Space with a Docker app that serves the Leaflet viewer,
    /// and stores tiles in a sibling bucket auto-named `<namespace>/<repo>_bucket`.
    ///
    /// Contrast with `--tiles hf://...`, which uploads only the viewer bundle
    /// (no Space scaffolding). Combine with `--tiles <local_dir>` and no input
    /// files to re-deploy an already-rendered tile directory without re-rendering.
    #[arg(long, conflicts_with = "output")]
    space: Option<String>,

    /// Regenerate index.html for an existing tiles directory without re-rendering tiles
    #[arg(long, value_name = "TILES_DIR", conflicts_with_all = ["files", "diff", "output", "tiles", "space"])]
    regen_html: Option<PathBuf>,

    /// Title shown in the HTML info panel (default: "arbvis" or "arbvis diff")
    #[arg(long, value_name = "TITLE")]
    title: Option<String>,

    /// Color regions by xorb ID for xet-backed files; hue = xorb, intensity = byte.
    #[arg(long)]
    show_xet_xorbs: bool,

    /// Tile output format. AVIF (default) is ~30-50% smaller than PNG over
    /// the wire; PNG is the universal fallback.
    #[arg(long, value_enum, default_value_t = TileFormatArg::Avif)]
    tile_format: TileFormatArg,

    /// Opt in to streaming I/O. Keeps `hf://` inputs remote (per-tile range
    /// fetches instead of an up-front download) and — when combined with an
    /// `hf://` output or `--space` — pushes tiles to the Hub as they're
    /// produced rather than staging through a local tempdir. Off by default;
    /// the disk-backed path is much faster and more recoverable. Use
    /// `--stream` when input or output data won't fit on local disk.
    ///
    /// Applies to every flow that takes `hf://` inputs: the normal renderer,
    /// `--diff` (when both sides are repo-level URLs), and `--moe-diff`.
    /// Single-file / local-path inputs always resolve through
    /// `hf_url::resolve` + mmap and are unaffected by `--stream`.
    #[arg(long)]
    stream: bool,
}

/// Bag of parameters shared by every render entrypoint. Avoids the
/// repeated-argument-list-of-doom that the call sites had before.
struct RenderConfig {
    /// Display title for the viewer / single-image label. `Cow` so the
    /// common default ("arbvis" / "arbvis diff" / "arbvis moe-diff") stays
    /// as a `&'static str` borrow instead of allocating on every run.
    title: Cow<'static, str>,
    inputs: Vec<String>,
    diff_mode: bool,
    show_xet_xorbs: bool,
    layout_mode: LayoutMode,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
}

/// Where the render output goes after rendering. Owns any temporary
/// directories so they live until the upload step completes.
///
/// `_tempdir: Option<TempDir>` is a **named** binding-with-leading-underscore,
/// NOT a wildcard pattern. The TempDir's `Drop` impl removes the directory
/// from disk; we need the binding alive until the post-render upload reads
/// from `local`. Rust drops named bindings at end-of-scope (in
/// `dispatch_render`'s match arms that is after the `.await` returns), which
/// is what keeps the upload reading a real path. If a future refactor renames
/// `_tempdir` to plain `_`, that becomes a wildcard pattern that drops
/// IMMEDIATELY at the destructure point — the directory is gone before the
/// upload starts. Do not rename.
enum OutputDest {
    /// No `--output`, no `--tiles`, no `--space`: pop a minifb window.
    Window,
    /// `--output <path>`: a single PNG. If `upload_hf` is `Some`, `local` is a
    /// tempdir path that will be uploaded post-render.
    SingleImage {
        local: PathBuf,
        upload_hf: Option<String>,
        _tempdir: Option<TempDir>,
    },
    /// `--tiles <dir>` and/or `--space`: a tile pyramid.
    ///
    /// `local` is the disk path the pyramid renders into (a user dir, or a
    /// tempdir inside `_tempdir`). It is `None` when `--stream` is set and
    /// the destination is HF-bound — in that case the streaming path
    /// uploads tiles as they are produced and never touches local disk.
    Tiles {
        local: Option<PathBuf>,
        upload_hf: Option<String>,
        space: Option<String>,
        _tempdir: Option<TempDir>,
    },
}

impl OutputDest {
    /// Resolve the user's `--output`/`--tiles`/`--space` flags into one of
    /// the three concrete destinations.
    ///
    /// Tempdirs are allocated lazily: only when the disk-backed path will
    /// actually use one (`hf://` output without `--stream`). With `--stream`,
    /// streaming destinations skip the tempdir entirely so a read-only or
    /// full `/tmp` doesn't kill the run before it starts — which is exactly
    /// the environment `--stream` exists for.
    fn from_args(args: &Args) -> anyhow::Result<Self> {
        // `--output` always wins (clap forbids combining it with --tiles/--space).
        // `--stream` does not apply to single-image output — there's no
        // pyramid to stream, so we still go through a local PNG and upload it.
        if let Some(ref out) = args.output {
            return if hf_url::is_hf_path(out) {
                let td = tempfile::tempdir().context("creating output tempdir")?;
                let local = td.path().join("output.png");
                Ok(OutputDest::SingleImage {
                    local,
                    upload_hf: Some(out.to_string_lossy().into_owned()),
                    _tempdir: Some(td),
                })
            } else {
                Ok(OutputDest::SingleImage {
                    local: out.clone(),
                    upload_hf: None,
                    _tempdir: None,
                })
            };
        }

        let tiles_set = args.tiles.is_some();
        let space_set = args.space.is_some();
        if !tiles_set && !space_set {
            return Ok(OutputDest::Window);
        }

        // Reject `--tiles hf://X --space S`: the two flags are documented
        // alternatives (see `--space` help: "Contrast with --tiles hf://..."),
        // and silently picking one over the other would mean the same flags
        // produce different end-states under `--stream` vs. disk-backed.
        // `--space` + `--tiles <local_dir>` is fine and used by the
        // deploy-only shortcut for re-deploys.
        if let (Some(p), true) = (args.tiles.as_ref(), space_set) {
            if hf_url::is_hf_path(p) {
                anyhow::bail!(
                    "--tiles hf://… and --space are alternatives, not stackable: \
                     --space deploys via its own bucket; --tiles hf:// uploads to \
                     a separate repo. Pass one, or combine --space with --tiles \
                     <local_dir> to (re-)deploy an already-rendered pyramid."
                );
            }
        }

        // Streaming destinations skip the local pyramid entirely.
        let needs_staging = !args.stream;

        let (local, upload_hf, tempdir) = match &args.tiles {
            Some(p) => {
                if hf_url::is_hf_path(p) {
                    let url = p.to_string_lossy().into_owned();
                    if needs_staging {
                        let td = tempfile::tempdir().context("creating tiles tempdir")?;
                        (Some(td.path().to_path_buf()), Some(url), Some(td))
                    } else {
                        (None, Some(url), None)
                    }
                } else {
                    // Local --tiles <dir>: always disk-backed.
                    (Some(p.clone()), None, None)
                }
            }
            None => {
                // --space without --tiles: tempdir for disk-backed sync;
                // nothing for streaming (uploads go through the bucket sink).
                if needs_staging {
                    let td = tempfile::tempdir().context("creating tiles tempdir")?;
                    (Some(td.path().to_path_buf()), None, Some(td))
                } else {
                    (None, None, None)
                }
            }
        };

        Ok(OutputDest::Tiles {
            local,
            upload_hf,
            space: args.space.clone(),
            _tempdir: tempdir,
        })
    }
}

/// Drives the full arbvis pipeline (CLI → sources → layout → tiles/single).
///
/// `args` carries the byte-only CLI surface (input/output paths, `--diff`,
/// `--space`, etc.). `opts` carries the four model-side knobs that used to
/// live on `Args` directly but are now owned by `modelweightvis::Args` —
/// pure-arbvis callers pass [`ModelOpts::default`] and never see those
/// flags in their `--help`. `registry` carries the pluggable surface
/// (formats, layouts, diff builders, leaf renderers, plus the option-slot
/// hooks). The arbvis binary builds `Registry::with_defaults()`;
/// modelweightvis's binary extends it with tensor-aware plugins via
/// `modelweightvis::register_all`.
pub async fn run(args: Args, opts: ModelOpts, registry: registry::Registry) -> anyhow::Result<()> {
    if let Some(ref tile_dir) = args.regen_html {
        return tiled::regen_html(tile_dir);
    }

    if args.show_xet_xorbs && args.diff.is_some() {
        anyhow::bail!("--show-xet-xorbs is incompatible with --diff");
    }

    let dest = OutputDest::from_args(&args)?;

    let (leaf_format, pyramid_format) = args.tile_format.split();
    let layout_mode = opts.layout_mode;
    let stream = args.stream;
    let show_xet_xorbs = args.show_xet_xorbs;

    // --- MoE diff ---------------------------------------------------------
    if let Some(moe_arg) = opts.moe_diff {
        let hook = registry.moe_diff.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--moe-diff requires a tensor-aware backend (no `MoeDiffPrep` registered); \
                 use `modelweightvis` instead of `arbvis`"
            )
        })?;
        let metric = opts.diff_metric;
        let input = moe_arg.to_string_lossy().into_owned();
        let inputs = vec![input.clone()];
        let title = default_title(args.title, "arbvis moe-diff");
        let (sources, total) = hook
            .prepare(&input, metric, stream)
            .await
            .with_context(|| format!("--moe-diff {input}"))?;
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        let cfg = RenderConfig {
            title,
            inputs,
            diff_mode: true,
            show_xet_xorbs: false,
            layout_mode,
            leaf_format,
            pyramid_format,
        };
        return dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await;
    }

    // --- MoE summary ------------------------------------------------------
    if let Some(summary_arg) = opts.moe_summary {
        let hook = registry.moe_summary.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--moe-summary requires a tensor-aware backend (no `MoeSummaryPrep` registered); \
                 use `modelweightvis` instead of `arbvis`"
            )
        })?;
        let stat = opts.summary_stat;
        let input = summary_arg.to_string_lossy().into_owned();
        let inputs = vec![input.clone()];
        let title = default_title(args.title, "arbvis moe-summary");
        let (sources, total) = hook
            .prepare(&input, stat, stream)
            .await
            .with_context(|| format!("--moe-summary {input}"))?;
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        // Not a true diff (no orig/mod sides) — `diff_mode = false` so layout
        // plugins apply their strict applicability gates. The summary layout
        // plugin signals applicability through its own per-source extension
        // tag, not through diff_mode.
        let cfg = RenderConfig {
            title,
            inputs,
            diff_mode: false,
            show_xet_xorbs: false,
            layout_mode,
            leaf_format,
            pyramid_format,
        };
        return dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await;
    }

    // --- MoE CKA ----------------------------------------------------------
    if let Some(cka_arg) = opts.moe_cka {
        let hook = registry.moe_cka.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "--moe-cka requires a tensor-aware backend (no `MoeCkaPrep` registered); \
                 use `modelweightvis` instead of `arbvis`"
            )
        })?;
        let sample = opts.cka_sample;
        let input = cka_arg.to_string_lossy().into_owned();
        let inputs = vec![input.clone()];
        let title = default_title(args.title, "arbvis moe-cka");
        let (sources, total) = hook
            .prepare(&input, sample, stream)
            .await
            .with_context(|| format!("--moe-cka {input}"))?;
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        // Same diff_mode reasoning as --moe-summary above.
        let cfg = RenderConfig {
            title,
            inputs,
            diff_mode: false,
            show_xet_xorbs: false,
            layout_mode,
            leaf_format,
            pyramid_format,
        };
        return dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await;
    }

    // --- Two-input diff ---------------------------------------------------
    if let Some(raw_diff_args) = args.diff {
        let diff_input_strs: Vec<String> = raw_diff_args
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let diff_title = default_title(args.title, "arbvis diff");
        let orig_str = &diff_input_strs[0];
        let mod_str = &diff_input_strs[1];
        let is_finetune = resolve_finetune(
            opts.finetune,
            opts.no_finetune,
            orig_str,
            mod_str,
            &registry,
        )
        .await;
        let metric = opts.diff_metric;

        let (sources, total) = if hf_url::is_repo_level(orig_str)?
            && hf_url::is_repo_level(mod_str)?
        {
            // Both are repo-level hf:// URLs: list files over API, diff lazily
            // over HTTP. Tensor-aware prep is required because the
            // model-format files need byte-range header parses to match
            // tensors across the two repos — arbvis core delegates to the
            // registered `RepoDiffPrep` hook.
            let hook = registry.repo_diff.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--diff between two `hf://` model repos requires a tensor-aware \
                     backend (no `RepoDiffPrep` registered); use `modelweightvis` \
                     instead of `arbvis`"
                )
            })?;
            let (orig_specs, mod_specs) = tokio::try_join!(
                async {
                    hf_url::list_repo_as_http_specs(orig_str)
                        .await
                        .with_context(|| format!("listing files in {orig_str}"))
                },
                async {
                    hf_url::list_repo_as_http_specs(mod_str)
                        .await
                        .with_context(|| format!("listing files in {mod_str}"))
                },
            )?;
            hook.prepare(&orig_specs, &mod_specs, is_finetune, metric, stream)
                .await?
        } else {
            // At least one side is a local path or single-file hf:// URL.
            // Resolve the two sides concurrently; order matters for the
            // (orig, modified) contract, so use `buffered` not
            // `buffer_unordered`.
            let diff_args: Vec<PathBuf> =
                futures::stream::iter(raw_diff_args.into_iter().map(resolve_input))
                    .buffered(2)
                    .try_collect()
                    .await?;
            data::prepare_diff_sources(&diff_args[0], &diff_args[1], is_finetune, metric, &registry)
                .await?
        };
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        let cfg = RenderConfig {
            title: diff_title,
            inputs: diff_input_strs,
            diff_mode: true,
            show_xet_xorbs: false,
            layout_mode,
            leaf_format,
            pyramid_format,
        };
        return dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await;
    }

    // --- Normal flow ------------------------------------------------------
    // Deploy-only shortcut: `--space` + `--tiles <local>` with no input files
    // means the tiles directory is already fully rendered; just deploy it.
    //
    // The match is the SAME shape as the rest of the dispatch — only the
    // `local: Some(p)` (user-provided dir) + `_tempdir: None` (we didn't
    // allocate one) combo distinguishes 'real on-disk pyramid the user wants
    // re-deployed' from 'tempdir-staged pyramid currently being rendered'.
    // If `OutputDest::from_args` is ever changed so that `--tiles <local>` +
    // `--space` allocates a tempdir, this shortcut silently stops firing and
    // we re-render from empty stdin — the regression the original comment
    // explicitly warns about (`labels.json` overwritten with a useless
    // `stdin` entry).
    if args.files.is_empty() && args.file_list.is_none() {
        if let OutputDest::Tiles {
            local: Some(local),
            upload_hf: None,
            space: Some(space_id),
            _tempdir: None,
        } = &dest
        {
            return deploy::run_deploy(local, space_id).await;
        }
    }

    let files = collect_input_files(args.files, args.file_list)?;
    let original_inputs: Vec<String> = files
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let (sources, total) = resolve_input_sources(&files, show_xet_xorbs, stream, &registry).await?;
    let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
    let cfg = RenderConfig {
        title: default_title(args.title, "arbvis"),
        inputs: original_inputs,
        diff_mode: false,
        show_xet_xorbs,
        layout_mode,
        leaf_format,
        pyramid_format,
    };
    dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await
}

/// Pick the viewer title: the user's `--title` if set, else a `&'static`
/// default. `Cow` keeps the default zero-alloc.
fn default_title(user: Option<String>, fallback: &'static str) -> Cow<'static, str> {
    user.map_or(Cow::Borrowed(fallback), Cow::Owned)
}

async fn resolve_finetune(
    finetune: bool,
    no_finetune: bool,
    orig_str: &str,
    mod_str: &str,
    registry: &registry::Registry,
) -> bool {
    if finetune {
        log::info!("--diff finetune mode: forced on by --finetune");
        return true;
    }
    if no_finetune {
        log::info!("--diff finetune mode: forced off by --no-finetune");
        return false;
    }
    let detected = match registry.finetune_detect.as_ref() {
        Some(hook) => hook.detect(orig_str, mod_str).await,
        None => None,
    };
    match detected {
        Some(true) => {
            log::info!(
                "--diff finetune mode: auto-detected ON ({} declares {} as its base in its HF model card)",
                mod_str, orig_str
            );
            true
        }
        Some(false) => {
            log::info!(
                "--diff finetune mode: auto-detected OFF ({} does not declare {} as a finetune base)",
                mod_str, orig_str
            );
            false
        }
        None => {
            log::info!(
                "--diff finetune mode: auto-detect skipped (no `FinetuneDetect` registered, not both hf:// model URLs, or API lookup failed); defaulting to OFF — pass --finetune to override"
            );
            false
        }
    }
}

/// Read `--files` and `--file-list` into a single flat path list.
fn collect_input_files(
    files: Vec<PathBuf>,
    file_list: Option<PathBuf>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = files;
    if let Some(list_path) = file_list {
        let reader: Box<dyn Read> = if list_path.as_os_str() == "-" {
            Box::new(io::stdin())
        } else {
            Box::new(
                File::open(&list_path)
                    .with_context(|| format!("failed to open {}", list_path.display()))?,
            )
        };
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                out.push(PathBuf::from(trimmed));
            }
        }
    }
    Ok(out)
}

/// Build `Source`s for the normal (non-diff) flow.
///
/// In disk-backed mode (the default), every `hf://` input is downloaded to
/// the local HF cache (via the `hf` CLI) by [`data::materialize_http_sources`]. In
/// `--stream` mode the sources stay remote and per-tile reads hit HTTP
/// directly; that's only useful when inputs don't fit on local disk and is
/// substantially slower otherwise. `--show-xet-xorbs` captures xet term
/// metadata before materialization (the post-download file lacks the remote
/// spec needed to query it).
async fn resolve_input_sources(
    files: &[PathBuf],
    show_xet_xorbs: bool,
    stream: bool,
    registry: &registry::Registry,
) -> anyhow::Result<(Vec<Source>, u64)> {
    // Streaming mode keeps hf:// inputs remote; show-xet-xorbs is the other
    // case that has to start from specs (it needs the remote spec to look up
    // xet metadata). Otherwise we can fast-path through `prepare_sources`
    // which downloads via hf_url::resolve and mmaps the local file.
    if !stream && !show_xet_xorbs {
        let resolved: Vec<PathBuf> =
            futures::stream::iter(files.iter().cloned().map(resolve_input))
                .buffered(RESOLVE_CONCURRENCY)
                .try_collect()
                .await?;
        return data::prepare_sources(&resolved, registry);
    }

    // Resolve `hf://` paths into specs concurrently. Repo-level URLs expand
    // to multiple specs; non-hf:// paths stay as `InputSpec::Local`. The
    // result preserves input order so labels and byte offsets are
    // deterministic across runs.
    let specs: Vec<InputSpec> = futures::stream::iter(files.iter().cloned().map(|p| async move {
        let s = p.to_string_lossy();
        if hf_url::is_hf_url(&s) {
            if hf_url::is_repo_level(&s)? {
                let listed = hf_url::list_repo_as_http_specs(&s)
                    .await
                    .with_context(|| format!("listing files in {s}"))?;
                anyhow::Ok(
                    listed
                        .into_iter()
                        .map(|(_, spec)| InputSpec::Remote(spec))
                        .collect::<Vec<_>>(),
                )
            } else {
                anyhow::Ok(vec![InputSpec::Remote(hf_url::resolve_to_http(&p).await?)])
            }
        } else {
            anyhow::Ok(vec![InputSpec::Local(p)])
        }
    }))
    .buffered(RESOLVE_CONCURRENCY)
    .try_collect::<Vec<_>>()
    .await?
    .into_iter()
    .flatten()
    .collect();
    let (mut sources, total) = data::prepare_sources_from_specs(&specs, registry).await?;
    if show_xet_xorbs {
        data::populate_xet_terms(&mut sources).await?;
    }
    if !stream {
        // Per-range HTTPS GETs are too expensive for the tile workload
        // — one whole-file download amortises the connection setup over the
        // entire file (which we read every byte of anyway during render).
        // See `materialize_http_sources` for the full story.
        data::materialize_http_sources(&mut sources).await?;
    }
    Ok((sources, total))
}

/// Drive the renderer for one of the four output destinations, optionally
/// through the streaming path. Centralises the cascade that used to be
/// duplicated three times in `run()`.
///
/// All three preparation paths (normal `resolve_input_sources`, the file-
/// pair / repo-level / moe `--diff`*, and `--moe-diff`) funnel through here,
/// which makes this the natural single place to run the registry's
/// [`PrepareSourcesExtension`] cross-source enrichment pass. Each path's
/// own preparer (e.g. `FormatPlugin::populate_*`) stuffs format-specific
/// per-source data into `Source.extensions`; the extension hook adds
/// cross-source / sidecar data (deduped by repo or parent dir) before any
/// layout selection sees the sources.
async fn dispatch_render(
    mut sources: Vec<Source>,
    total: u64,
    labels: &[PathBuf],
    cfg: &RenderConfig,
    dest: OutputDest,
    stream: bool,
    registry: &registry::Registry,
) -> anyhow::Result<()> {
    if let Some(hook) = registry.prepare_sources_extension.as_ref() {
        hook.enrich(&mut sources)
            .await
            .context("prepare_sources_extension hook failed")?;
    }
    match dest {
        OutputDest::Window => {
            run_single_blocking(labels.to_vec(), None, sources, total, cfg, registry).await?;
            Ok(())
        }
        OutputDest::SingleImage {
            local,
            upload_hf,
            _tempdir,
        } => {
            run_single_blocking(
                labels.to_vec(),
                Some(local.clone()),
                sources,
                total,
                cfg,
                registry,
            )
            .await?;
            if let Some(url) = upload_hf {
                deploy::upload_file_to(&url, &local).await?;
            }
            Ok(())
        }
        OutputDest::Tiles {
            local,
            upload_hf,
            space,
            _tempdir,
        } => {
            render_tiles(
                sources,
                total,
                local.as_deref(),
                upload_hf,
                space,
                cfg,
                stream,
                registry,
            )
            .await
        }
    }
}

/// Run the synchronous single-image pipeline on the tokio blocking pool so
/// the tokio runtime stays responsive for any async helpers (`Data::Xet`,
/// `Data::LazyDiff`) the renderer calls into.
async fn run_single_blocking(
    labels: Vec<PathBuf>,
    output: Option<PathBuf>,
    sources: Vec<Source>,
    total: u64,
    cfg: &RenderConfig,
    registry: &registry::Registry,
) -> anyhow::Result<()> {
    let diff_mode = cfg.diff_mode;
    let show_xet_xorbs = cfg.show_xet_xorbs;
    let layout_mode = cfg.layout_mode;
    // `spawn_blocking` requires a `'static` closure, so clone the registry
    // (cheap — every slot is an `Arc<…>` clone).
    let registry = registry.clone();
    tokio::task::spawn_blocking(move || {
        single::run_single(
            &labels,
            output,
            sources,
            total,
            diff_mode,
            show_xet_xorbs,
            layout_mode,
            &registry,
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("run_single join failure: {e}"))?
}

/// Tile-pyramid render + upload + Space-deploy fan-out.
///
/// When `stream` is true and the destination is HF-bound (`upload_hf` or
/// `space`), tiles are pushed directly to the Hub without staging through
/// `local` — the off-by-default streaming path. Local-only destinations
/// always go through disk-backed `run_tiles`, even with `--stream`; the
/// stream flag still cuts the input-side download (see
/// [`resolve_input_sources`]).
///
/// `local` is `Some(_)` for every disk-backed call; `None` only when
/// `OutputDest::from_args` skipped the tempdir because `--stream` was set
/// and the destination is HF-bound.
async fn render_tiles(
    sources: Vec<Source>,
    total: u64,
    local: Option<&Path>,
    upload_hf: Option<String>,
    space: Option<String>,
    cfg: &RenderConfig,
    stream: bool,
    registry: &registry::Registry,
) -> anyhow::Result<()> {
    let hf_destined = upload_hf.is_some() || space.is_some();
    if stream && hf_destined {
        return render_tiles_streaming(sources, total, upload_hf, space, cfg, registry).await;
    }

    let local = local.ok_or_else(|| {
        anyhow::anyhow!(
            "internal: disk-backed tile render without a local path \
             (OutputDest::from_args should have allocated one)"
        )
    })?;

    // Migration hint: large hf:// outputs that used to stream now stage
    // through a tempdir. Tell the user how to opt back into streaming so a
    // /tmp ENOSPC isn't the first signal of the changed default.
    if hf_destined {
        log::info!(
            "Disk-backed tile render: staging full pyramid in {} before upload. \
             Pass --stream to skip local staging when the pyramid won't fit on disk.",
            local.display()
        );
    }

    tiled::run_tiles(
        sources,
        total,
        local.to_path_buf(),
        cfg.diff_mode,
        &cfg.title,
        &cfg.inputs,
        cfg.show_xet_xorbs,
        cfg.leaf_format,
        cfg.pyramid_format,
        cfg.layout_mode,
        registry,
    )
    .await?;
    if let Some(ref space_id) = space {
        deploy::run_deploy(local, space_id).await?;
    }
    if let Some(ref url) = upload_hf {
        deploy::upload_dir_to(url, local).await?;
    }
    Ok(())
}

/// Streaming variant of [`render_tiles`]: push tiles to the Hub as they're
/// produced, no local pyramid staging. Off-by-default, gated behind
/// `--stream` *and* an HF destination.
async fn render_tiles_streaming(
    sources: Vec<Source>,
    total: u64,
    upload_hf: Option<String>,
    space: Option<String>,
    cfg: &RenderConfig,
    registry: &registry::Registry,
) -> anyhow::Result<()> {
    use crate::tiled::streaming::run_tiles_hf_streaming;

    // --space + --stream: render through the space's bucket and deploy the app.
    if let Some(space_id) = space {
        let bucket = deploy::create_space_bucket(&space_id).await?;
        let html = run_tiles_hf_streaming(
            sources,
            total,
            &bucket,
            cfg.diff_mode,
            &cfg.title,
            &cfg.inputs,
            cfg.show_xet_xorbs,
            cfg.leaf_format,
            cfg.pyramid_format,
            cfg.layout_mode,
            registry,
        )
        .await?;
        deploy::deploy_space_app(&space_id, &bucket.repo_id, html).await?;
        return Ok(());
    }

    // --tiles hf://… + --stream: push to the named repo, no Space.
    let hf_url = upload_hf
        .ok_or_else(|| anyhow::anyhow!("internal: stream dispatch with no HF destination"))?;
    let hf_out: HfOutputSpec = hf_url::parse_hf_output(&hf_url)?;
    run_tiles_hf_streaming(
        sources,
        total,
        &hf_out,
        cfg.diff_mode,
        &cfg.title,
        &cfg.inputs,
        cfg.show_xet_xorbs,
        cfg.leaf_format,
        cfg.pyramid_format,
        cfg.layout_mode,
        registry,
    )
    .await?;
    Ok(())
}

/// Resolve an input path: download from HF if it starts with `hf://`.
async fn resolve_input(path: PathBuf) -> anyhow::Result<PathBuf> {
    let display = path.display().to_string();
    hf_url::resolve(&path)
        .await
        .with_context(|| format!("resolving {display}"))
}

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
