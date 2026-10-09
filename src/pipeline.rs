//! The render pipeline: `run()` (CLI args → sources → render → deploy) and
//! the destination fan-out (`dispatch_render` and the disk-backed / streaming
//! render paths it centralises).

use std::borrow::Cow;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::Context;

use crate::cli::{
    check_bare_run_inputs, collect_input_files, default_title, ignored_3d_flags, validate_grid,
    validate_volume_res, volume_res_ignored_warning, Args, OutputDest,
};
use crate::data::Source;
use crate::deploy;
use crate::hf_url::{self, HfOutputSpec};
use crate::providers::select_provider;
use crate::registry::{self, DestKind, DiffPair, SourceCtx};
use crate::tiled;
use crate::tiled::streaming::run_tiles_hf_streaming;
use crate::volume;
use crate::TileFormat;

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
    layout_mode: crate::layout::LayoutMode,
    leaf_format: TileFormat,
    pyramid_format: TileFormat,
    /// `--3d`: route to the volume renderer instead of the tile pyramid.
    three_d: bool,
    /// 3D voxel grid side (power of two); unused in 2D.
    grid_side: u32,
    /// 3D volume virtual resolution for the sparse brick pool (`0` = `--grid`).
    volume_res: u32,
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
/// Source preparation is fully delegated to the registry's
/// [`registry::SourceProvider`]s: `run` builds a neutral [`SourceCtx`] from the
/// parsed args and picks the highest-priority applicable provider (the
/// `i32::MIN` floor always applies).
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
        validate_volume_res(args.volume_res)?;
        if let Some(msg) = volume_res_ignored_warning(args.grid, args.volume_res) {
            log::warn!("{msg}");
        }
        if args.show_xet_xorbs {
            log::warn!("--show-xet-xorbs has no effect in --3d mode; ignoring");
        }
    } else {
        let ignored = ignored_3d_flags(args.grid, args.volume_res);
        if !ignored.is_empty() {
            log::warn!(
                "{} only take effect with --3d; ignoring them in 2D mode",
                ignored.join(", ")
            );
        }
    }

    let dest = if args.png.is_some() {
        // Single-PNG mode writes one file, not a bundle: skip --out/--space
        // destination resolution entirely (`--out DIR` is re-read directly as
        // the PNG's parent directory by `png_output_path`).
        None
    } else {
        Some(OutputDest::from_args(&args)?)
    };

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
        if let Some(OutputDest::Bundle {
            local: Some(local),
            upload_hf: None,
            space: Some(space_id),
            _tempdir: None,
        }) = &dest
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
    // Guard the stdin fallback: with no files, the byte provider reads stdin
    // to EOF, which blocks forever on an interactive terminal (see
    // `check_bare_run_inputs`). Piped/redirected stdin is unaffected.
    check_bare_run_inputs(files.is_empty(), std::io::stdin().is_terminal())?;
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
        dest_kind: dest.as_ref().map(|d| d.kind()).unwrap_or(DestKind::Bundle),
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
        // The requested detail resolution; `render_volume` splits it into a
        // small coarse dense grid + a streamed brick pool (byte path only).
        grid_side: args.grid,
        volume_res: args.volume_res,
    };

    // Single-image PNG export: render the whole 2D canvas as one PNG and
    // stop — no bundle, no upload, no deploy. Diff mode renders through the
    // signed-delta LUT + crosshatch (truecolor); plain mode is indexed.
    if let Some(ref png) = args.png {
        if total == 0 {
            anyhow::bail!("--png requires non-empty input");
        }
        let out_path = tiled::single::png_output_path(png, args.out.as_deref())?;
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        return if hints.diff_mode {
            tiled::single::render_single_diff_png(&sources, total, &out_path).await
        } else {
            tiled::single::render_single_png(&sources, total, &out_path).await
        };
    }

    match dest {
        Some(dest) => dispatch_render(sources, total, &labels, &cfg, dest, stream, &registry).await,
        // PNG mode returned earlier; nothing else runs without a destination.
        None => Ok(()),
    }
}

/// Drive the renderer for one of the four output destinations, optionally
/// through the streaming path. Centralises the cascade that used to be
/// duplicated three times in `run()`.
///
/// All preparation paths (normal `resolve_input_sources`, the file-pair /
/// repo-level `--diff`, and the MoE-mode preps) funnel through here,
/// which makes this the natural single place to run the registry's
/// [`registry::PrepareSourcesExtension`] cross-source enrichment pass. Each path's
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
        cfg.volume_res,
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
/// [`crate::providers`]'s `resolve_input_sources`).
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
