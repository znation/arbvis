#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod color;
pub mod data;
mod deploy;
mod geometry;
pub mod hf_cli;
mod hf_upload;
pub mod hf_url;
mod json_diff;
mod layout;
mod perf_monitor;
mod progress;
mod registry;
mod throttle;
mod tiled;
mod volume;
pub mod xet;

// Public library surface — the byte-only foundation modelweightvis builds
// on. Tile pipeline, source/diff plumbing, layout traits, hooks for the
// model-aware plugins to plug into.
pub use data::{
    byte_directory_diff, load_source_data, prepare_diff_sources, prepare_sources,
    prepare_sources_from_specs, CustomSource, Data, DiffFill, Extensions, LazyFetcher, SceneTag,
    Source, SourceKind,
};
pub use geometry::name_hue;
pub use layout::{CanvasGeom, LayoutMode, LayoutShape};
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
    VolumeEntity, VolumeShape, VoxelBox, VoxelCell, VoxelGridMut, VoxelRegistry, VoxelRenderCtx,
    VoxelRenderer,
};

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use futures::stream::{StreamExt, TryStreamExt};
use tempfile::TempDir;

use crate::data::InputSpec;
use crate::hf_url::HfOutputSpec;

/// Concurrency cap when resolving (downloading) `hf://` inputs at startup.
/// Mirrors `data::SETUP_FETCH_CONCURRENCY` so user-visible parallelism stays
/// consistent across the input-resolution and materialisation stages.
const RESOLVE_CONCURRENCY: usize = 16;

// arbvis's two built-in [`SourceProvider`]s, installed by
// [`Registry::with_defaults`]. A specialization registers its own
// higher-priority providers; these are the always-present byte fallbacks.

/// Byte/JSON diff over a `--diff` pair (priority 100). Resolves both sides
/// (local path or single-file `hf://`) and dispatches through the file-pair
/// builder cascade or [`data::byte_directory_diff`].
struct ByteDiffProvider;

#[async_trait(?Send)]
impl SourceProvider for ByteDiffProvider {
    fn id(&self) -> &'static str {
        "byte-diff"
    }
    fn priority(&self) -> i32 {
        100
    }
    fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
        ctx.diff.is_some()
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        let diff = ctx
            .diff
            .as_ref()
            .expect("ByteDiffProvider applies only when --diff is set");
        // Resolve both sides concurrently; the (original, modified) order is
        // part of the diff contract, so `try_join!` (not unordered) is used.
        let (orig, mod_) = tokio::try_join!(
            resolve_input(PathBuf::from(diff.original)),
            resolve_input(PathBuf::from(diff.modified)),
        )?;
        let (sources, total) =
            data::prepare_diff_sources(&orig, &mod_, false, ctx.registry).await?;
        let hints = RenderHints {
            diff_mode: true,
            title_suffix: Cow::Borrowed("diff"),
            show_xet_xorbs: false,
            inputs: vec![diff.original.to_string(), diff.modified.to_string()],
        };
        Ok((sources, total, hints))
    }
}

/// Normal byte render of the positional inputs (or stdin). The priority floor
/// (`i32::MIN`) — always applicable, so provider selection always terminates.
struct NormalBytesProvider;

#[async_trait(?Send)]
impl SourceProvider for NormalBytesProvider {
    fn id(&self) -> &'static str {
        "normal-bytes"
    }
    fn priority(&self) -> i32 {
        i32::MIN
    }
    fn applicable(&self, _ctx: &SourceCtx<'_>) -> bool {
        true
    }
    async fn prepare(
        &self,
        ctx: &SourceCtx<'_>,
    ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
        let (sources, total) =
            resolve_input_sources(ctx.inputs, ctx.show_xet_xorbs, ctx.stream, ctx.registry).await?;
        let inputs: Vec<String> = ctx
            .inputs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let hints = RenderHints {
            diff_mode: false,
            title_suffix: Cow::Borrowed(""),
            show_xet_xorbs: ctx.show_xet_xorbs,
            inputs,
        };
        Ok((sources, total, hints))
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
                // Leaf speed 8 (was 6) to match the pyramid preset. Only bites
                // for Xet-mode leaves, which are the sole AVIF leaf tiles —
                // Plain/Dtype leaves are indexed PNG (see `TileFormat` docs). For
                // those AVIF leaves it trims rav1e's rate-distortion search
                // (`rdo_mode_decision`/`rdo_partition_decision`, the dominant cost
                // in the encode profile) for a modest size bump; at quality 100
                // the output stays near-lossless regardless of speed.
                TileFormat::Avif {
                    quality: 100,
                    speed: 8,
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

    /// Write the viewer bundle to this directory. In the default 2D mode this is
    /// a Leaflet tile pyramid (`tiles/`, `index.html`, `labels.json`); under
    /// `--3d` it is the Three.js volume bundle (`index.html`, `volume.bin`,
    /// `points.bin`, `meta.json`). Open `index.html` over HTTP in a browser.
    ///
    /// Accepts a local directory or an `hf://` URL to upload the bundle to a Hub
    /// repo. Note: `hf://` upload does NOT stand up a Space; the `index.html`
    /// lands in the target repo but won't render on the Hub on its own. Use
    /// `--space` for a working visualization URL.
    #[arg(short = 'o', long)]
    out: Option<PathBuf>,

    /// Render in 3D: lay bytes along a 3D Hilbert curve in a cube and emit a
    /// Three.js viewer (volume + point-cloud modes) instead of the 2D tile
    /// pyramid. Opacity encodes density so the cube's interior is visible.
    #[arg(long = "3d")]
    three_d: bool,

    /// 3D voxel grid side (a power of two, 2–512). Higher is more detailed but
    /// a larger download (≈ side³·4 bytes). Ignored in 2D mode.
    #[arg(long, default_value_t = 128)]
    grid: u32,

    /// Visualize abs(modified - original) byte differences; ORIGINAL and MODIFIED are files or directories
    #[arg(long, num_args = 2, value_names = ["ORIGINAL", "MODIFIED"])]
    diff: Option<Vec<PathBuf>>,

    // Specialization-specific flags (a multi-scene view, a structured layout
    // choice, format-specific diff knobs, etc.) live on the *downstream*
    // crate's clap struct, which flattens this one. The downstream reads those
    // flags to construct its own `SourceProvider`s / set `Registry::layout_mode`
    // before calling `run`; arbvis-only callers never see them in `arbvis --help`.
    /// Render and deploy a viewable HF Space (e.g. username/my-vis). Creates the
    /// Space with a Docker app that serves the viewer, and stores the bundle in a
    /// sibling bucket auto-named `<namespace>/<repo>_bucket`. Works for both 2D
    /// and `--3d`.
    ///
    /// Contrast with `--out hf://...`, which uploads only the viewer bundle
    /// (no Space scaffolding). Combine with `--out <local_dir>` and no input
    /// files to re-deploy an already-rendered bundle without re-rendering.
    #[arg(long)]
    space: Option<String>,

    /// Regenerate index.html for an existing bundle directory without re-rendering
    #[arg(long, value_name = "DIR", conflicts_with_all = ["files", "diff", "out", "space"])]
    regen_html: Option<PathBuf>,

    /// Title shown in the HTML info panel (default: the brand name, optionally
    /// suffixed " diff" / " moe" — e.g. "arbvis" / "arbvis moe")
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
    /// `--diff` (when both sides are repo-level URLs), and the MoE viewer
    /// (`--moe`).
    /// Single-file / local-path inputs always resolve through
    /// `hf_url::resolve` + mmap and are unaffected by `--stream`.
    #[arg(long)]
    stream: bool,
}

/// Bag of parameters shared by every render entrypoint. Avoids the
/// repeated-argument-list-of-doom that the call sites had before.
struct RenderConfig {
    /// Display title for the viewer / single-image label. Either the user's
    /// `--title` or the brand-derived default (`"{name}"` / `"{name} diff"` /
    /// `"{name} moe"`); see [`default_title`]. `Cow` avoids a clone when the
    /// caller already owns the string.
    title: Cow<'static, str>,
    inputs: Vec<String>,
    diff_mode: bool,
    show_xet_xorbs: bool,
    layout_mode: LayoutMode,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    /// `--3d`: route to the volume renderer instead of the tile pyramid.
    three_d: bool,
    /// 3D voxel grid side (power of two); unused in 2D.
    grid_side: u32,
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
    /// `--out <dir>` and/or `--space`: a web-viewer bundle directory (a 2D tile
    /// pyramid, or — under `--3d` — a volume bundle).
    ///
    /// `local` is the disk path the bundle renders into (a user dir, or a
    /// tempdir inside `_tempdir`). It is `None` only for the 2D streaming path
    /// (`--stream` + HF destination), which pushes tiles as produced and never
    /// touches local disk; 3D always stages locally.
    Bundle {
        local: Option<PathBuf>,
        upload_hf: Option<String>,
        space: Option<String>,
        _tempdir: Option<TempDir>,
    },
}

impl OutputDest {
    /// Resolve the user's `--out`/`--space` flags into the bundle destination.
    ///
    /// Tempdirs are allocated lazily: only when the disk-backed path will
    /// actually use one (`hf://` output without `--stream`, or any `--3d` run,
    /// which always stages locally). With 2D `--stream`, streaming destinations
    /// skip the tempdir entirely so a read-only or full `/tmp` doesn't kill the
    /// run before it starts — which is exactly the environment `--stream`
    /// exists for.
    fn from_args(args: &Args) -> anyhow::Result<Self> {
        let out_set = args.out.is_some();
        let space_set = args.space.is_some();
        if !out_set && !space_set {
            anyhow::bail!(
                "no output specified: pass --out <DIR> to write the viewer bundle \
                 (a local dir or an hf:// URL), or --space <NAMESPACE/REPO> to \
                 render and deploy a Hugging Face Space."
            );
        }

        // Reject `--out hf://X --space S`: the two flags are documented
        // alternatives (see `--space` help: "Contrast with --out hf://..."),
        // and silently picking one over the other would mean the same flags
        // produce different end-states. `--space` + `--out <local_dir>` is fine
        // and used by the deploy-only shortcut for re-deploys.
        if let (Some(p), true) = (args.out.as_ref(), space_set) {
            if hf_url::is_hf_path(p) {
                anyhow::bail!(
                    "--out hf://… and --space are alternatives, not stackable: \
                     --space deploys via its own bucket; --out hf:// uploads to \
                     a separate repo. Pass one, or combine --space with --out \
                     <local_dir> to (re-)deploy an already-rendered bundle."
                );
            }
        }

        // 3D always stages to a local dir (no streaming output path); 2D skips
        // staging only under --stream to an HF destination.
        let needs_staging = !args.stream || args.three_d;

        let (local, upload_hf, tempdir) = match &args.out {
            Some(p) => {
                if hf_url::is_hf_path(p) {
                    let url = p.to_string_lossy().into_owned();
                    if needs_staging {
                        let td = tempfile::tempdir().context("creating output tempdir")?;
                        (Some(td.path().to_path_buf()), Some(url), Some(td))
                    } else {
                        (None, Some(url), None)
                    }
                } else {
                    // Local --out <dir>: always disk-backed.
                    (Some(p.clone()), None, None)
                }
            }
            None => {
                // --space without --out: tempdir for disk-backed sync;
                // nothing for streaming (uploads go through the bucket sink).
                if needs_staging {
                    let td = tempfile::tempdir().context("creating output tempdir")?;
                    (Some(td.path().to_path_buf()), None, Some(td))
                } else {
                    (None, None, None)
                }
            }
        };

        Ok(OutputDest::Bundle {
            local,
            upload_hf,
            space: args.space.clone(),
            _tempdir: tempdir,
        })
    }

    /// Coarse destination shape for [`SourceCtx`], hiding the tempdir / upload
    /// internals so a provider can gate on it without depending on them.
    fn kind(&self) -> DestKind {
        match self {
            OutputDest::Bundle { .. } => DestKind::Bundle,
        }
    }
}

/// Drives the full arbvis pipeline (CLI → sources → layout → tiles/single).
///
/// `args` carries the byte-only CLI surface (input/output paths, `--diff`,
/// `--space`, etc.). `registry` carries the pluggable surface (formats,
/// layouts, diff builders, leaf renderers, source providers, single-image
/// renderers, branding, layout mode). The arbvis binary builds
/// `Registry::with_defaults()`; a downstream specialization extends it (e.g.
/// `modelweightvis::register_all` plus its own provider registration) and maps
/// its own flags onto the registry before calling `run`.
///
/// Source preparation is fully delegated to the registry's [`SourceProvider`]s:
/// `run` builds a neutral [`SourceCtx`] from the parsed args and picks the
/// highest-priority applicable provider (the `i32::MIN` floor always applies).
pub async fn run(args: Args, registry: registry::Registry) -> anyhow::Result<()> {
    if let Some(ref dir) = args.regen_html {
        return if args.three_d {
            volume::regen_html(dir, &registry.branding)
        } else {
            tiled::regen_html(dir, &registry.branding)
        };
    }

    if args.show_xet_xorbs && args.diff.is_some() {
        anyhow::bail!("--show-xet-xorbs is incompatible with --diff");
    }
    if args.three_d {
        validate_grid(args.grid)?;
        if args.show_xet_xorbs {
            log::warn!("--show-xet-xorbs has no effect in --3d mode; ignoring");
        }
    }

    let dest = OutputDest::from_args(&args)?;

    let (leaf_format, pyramid_format) = args.tile_format.split();
    let stream = args.stream;
    let show_xet_xorbs = args.show_xet_xorbs;

    // Deploy-only shortcut (a destination concern, so it runs before any source
    // prep): `--space` + `--out <local>` with no input files means the bundle
    // directory is already fully rendered; just deploy it.
    //
    // The match relies on `local: Some(p)` (user-provided dir) + `_tempdir: None`
    // (we didn't allocate one) to distinguish 'real on-disk bundle the user
    // wants re-deployed' from 'tempdir-staged bundle currently being rendered'.
    // If `OutputDest::from_args` is ever changed so that `--out <local>` +
    // `--space` allocates a tempdir, this shortcut silently stops firing and
    // we re-render from empty stdin.
    if args.files.is_empty() && args.file_list.is_none() {
        if let OutputDest::Bundle {
            local: Some(local),
            upload_hf: None,
            space: Some(space_id),
            _tempdir: None,
        } = &dest
        {
            return if args.three_d {
                deploy::run_deploy_bundle(local, space_id).await
            } else {
                deploy::run_deploy(local, space_id).await
            };
        }
    }

    // Collect positional inputs (no stdin fallback here — the byte provider
    // reads stdin when its input list is empty). `--diff` sides are kept as an
    // ordered pair in the neutral `SourceCtx`.
    let files = collect_input_files(args.files, args.file_list)?;
    let diff_strs: Option<[String; 2]> = args.diff.as_ref().map(|v| {
        [
            v[0].to_string_lossy().into_owned(),
            v[1].to_string_lossy().into_owned(),
        ]
    });
    let diff = diff_strs.as_ref().map(|p| DiffPair {
        original: p[0].as_str(),
        modified: p[1].as_str(),
    });

    let ctx = SourceCtx {
        inputs: &files,
        diff,
        dest_kind: dest.kind(),
        three_d: args.three_d,
        stream,
        show_xet_xorbs,
        registry: &registry,
    };

    // Pick the highest-priority applicable provider. The `i32::MIN`
    // NormalBytesProvider floor always applies, so this always resolves —
    // mirrors `select_layout`.
    let chosen = select_provider(&registry.providers, &ctx)
        .expect("registry.providers must include the i32::MIN NormalBytesProvider floor");

    let (sources, total, hints) = chosen
        .prepare(&ctx)
        .await
        .with_context(|| format!("source provider `{}`", chosen.id()))?;

    let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
    let cfg = RenderConfig {
        title: default_title(args.title, &registry.branding.name, &hints.title_suffix),
        inputs: hints.inputs,
        diff_mode: hints.diff_mode,
        show_xet_xorbs: hints.show_xet_xorbs,
        layout_mode: registry.layout_mode,
        leaf_format,
        pyramid_format,
        three_d: args.three_d,
        grid_side: args.grid,
    };
    dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await
}

/// Validate `--grid`: a power of two in `[2, 512]`. (512³·4 ≈ 512 MiB on the
/// wire is already a lot; the lower bound keeps the Hilbert order ≥ 1.)
fn validate_grid(side: u32) -> anyhow::Result<()> {
    if !(2..=512).contains(&side) || !side.is_power_of_two() {
        anyhow::bail!("--grid must be a power of two between 2 and 512, got {side}");
    }
    Ok(())
}

/// Pick the highest-priority [`SourceProvider`] whose `applicable` returns true.
/// Sorted descending by priority; the `i32::MIN` floor (`NormalBytesProvider`)
/// guarantees a result for any registry built via [`Registry::with_defaults`].
fn select_provider<'a>(
    providers: &'a [Arc<dyn SourceProvider>],
    ctx: &SourceCtx<'_>,
) -> Option<&'a Arc<dyn SourceProvider>> {
    let mut sorted: Vec<&Arc<dyn SourceProvider>> = providers.iter().collect();
    sorted.sort_by_key(|p| std::cmp::Reverse(p.priority()));
    sorted.into_iter().find(|p| p.applicable(ctx))
}

/// Pick the viewer title: the user's `--title` if set, else the brand name
/// with a mode suffix (`"{name} moe"` / `"{name} diff"`, or just `"{name}"`
/// when `suffix` is empty). Built once per run, so the fallback allocation is
/// negligible.
fn default_title(user: Option<String>, name: &str, suffix: &str) -> Cow<'static, str> {
    match user {
        Some(t) => Cow::Owned(t),
        None if suffix.is_empty() => Cow::Owned(name.to_string()),
        None => Cow::Owned(format!("{name} {suffix}")),
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
/// All preparation paths (normal `resolve_input_sources`, the file-pair /
/// repo-level `--diff`, and the MoE-mode preps) funnel through here,
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
    let _ = labels; // labels feed the (2D-only) tile renderer's source names
    match dest {
        OutputDest::Bundle {
            local,
            upload_hf,
            space,
            _tempdir,
        } => {
            if cfg.three_d {
                render_volume_bundle(sources, total, local, upload_hf, space, cfg, registry).await
            } else {
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
}

/// 3D analog of [`render_tiles`]'s disk-backed path: render the volume bundle
/// into `local`, then upload / deploy. 3D always stages locally (no streaming
/// output), so `local` is always `Some` (see [`OutputDest::from_args`]).
async fn render_volume_bundle(
    sources: Vec<Source>,
    total: u64,
    local: Option<PathBuf>,
    upload_hf: Option<String>,
    space: Option<String>,
    cfg: &RenderConfig,
    registry: &registry::Registry,
) -> anyhow::Result<()> {
    let dir = local.ok_or_else(|| {
        anyhow::anyhow!(
            "internal: 3D render without a local staging dir \
             (OutputDest::from_args should have allocated one)"
        )
    })?;
    volume::render_volume(
        sources,
        total,
        dir.clone(),
        &cfg.title,
        &cfg.inputs,
        cfg.diff_mode,
        cfg.grid_side,
        cfg.layout_mode,
        registry,
        &registry.branding,
    )
    .await?;
    if let Some(ref space_id) = space {
        deploy::run_deploy_bundle(&dir, space_id).await?;
    }
    if let Some(ref url) = upload_hf {
        deploy::upload_dir_to(url, &dir).await?;
    }
    Ok(())
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

#[cfg(test)]
mod title_tests {
    use super::default_title;

    #[test]
    fn brand_and_suffix_fallbacks() {
        assert_eq!(
            default_title(None, "modelweightvis", "moe"),
            "modelweightvis moe"
        );
        assert_eq!(default_title(None, "arbvis", "diff"), "arbvis diff");
        assert_eq!(default_title(None, "arbvis", ""), "arbvis");
    }

    #[test]
    fn user_title_wins() {
        assert_eq!(
            default_title(Some("custom".to_string()), "arbvis", "moe"),
            "custom"
        );
    }
}

#[cfg(test)]
mod provider_selection_tests {
    use super::*;

    /// A high-priority diff-only provider, standing in for a downstream
    /// specialization's provider (e.g. modelweightvis's `RepoDiffProvider`).
    struct MockDiffProvider;

    #[async_trait(?Send)]
    impl SourceProvider for MockDiffProvider {
        fn id(&self) -> &'static str {
            "mock-high"
        }
        fn priority(&self) -> i32 {
            500
        }
        fn applicable(&self, ctx: &SourceCtx<'_>) -> bool {
            ctx.diff.is_some()
        }
        async fn prepare(
            &self,
            _ctx: &SourceCtx<'_>,
        ) -> anyhow::Result<(Vec<Source>, u64, RenderHints)> {
            Ok((Vec::new(), 0, RenderHints::default()))
        }
    }

    fn ctx<'a>(
        reg: &'a Registry,
        inputs: &'a [PathBuf],
        diff: Option<DiffPair<'a>>,
    ) -> SourceCtx<'a> {
        SourceCtx {
            inputs,
            diff,
            dest_kind: DestKind::Bundle,
            three_d: false,
            stream: false,
            show_xet_xorbs: false,
            registry: reg,
        }
    }

    // The byte built-ins: a normal-input invocation falls to the `i32::MIN`
    // floor; a `--diff` invocation is caught by the byte-diff provider.
    #[test]
    fn byte_builtins_ladder() {
        let reg = Registry::with_defaults();
        let inputs = vec![PathBuf::from("a.bin")];
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &inputs, None))
                .unwrap()
                .id(),
            "normal-bytes"
        );
        let none: Vec<PathBuf> = Vec::new();
        let pair = DiffPair {
            original: "a",
            modified: "b",
        };
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &none, Some(pair)))
                .unwrap()
                .id(),
            "byte-diff"
        );
    }

    // A higher-priority provider shadows the byte-diff built-in when it
    // applies, but the floor still wins when it doesn't.
    #[test]
    fn higher_priority_shadows_then_falls_through() {
        let mut reg = Registry::with_defaults();
        reg.providers.push(Arc::new(MockDiffProvider));
        let none: Vec<PathBuf> = Vec::new();
        let pair = DiffPair {
            original: "a",
            modified: "b",
        };
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &none, Some(pair)))
                .unwrap()
                .id(),
            "mock-high"
        );
        assert_eq!(
            select_provider(&reg.providers, &ctx(&reg, &none, None))
                .unwrap()
                .id(),
            "normal-bytes"
        );
    }
}
