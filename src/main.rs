mod color;
mod data;
mod deploy;
mod geometry;
mod hf_upload;
mod hf_url;
mod label;
mod progress;
mod safetensors;
mod single;
mod throttle;
mod tiled;
mod xet;

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use crate::data::InputSpec;

/// Visualize binary files as Hilbert curve plots.
///
/// Each byte is mapped to a color and placed along a Hilbert curve, so
/// structural patterns in the file (e.g. repeated null regions, ASCII text,
/// high-entropy compressed data) become visually apparent.
///
/// Reads from FILES if provided, otherwise reads from stdin.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Files to visualize (defaults to stdin); multiple files are concatenated
    #[arg(conflicts_with = "diff")]
    files: Vec<PathBuf>,

    /// Read file list from this file (one path per line), or - for stdin
    #[arg(short = 'l', long, conflicts_with = "diff")]
    file_list: Option<PathBuf>,

    /// Write the canvas to this PNG file instead of displaying a window
    #[arg(short, long, conflicts_with = "tiles")]
    output: Option<PathBuf>,

    /// Write a tiled pyramid to this directory for Leaflet.js viewing
    #[arg(short, long, conflicts_with = "output")]
    tiles: Option<PathBuf>,

    /// Sort bytes by value within each file before rendering (loads files into memory)
    #[arg(short = 's', long)]
    sort: bool,

    /// Visualize abs(modified - original) byte differences; ORIGINAL and MODIFIED are files or directories
    #[arg(long, num_args = 2, value_names = ["ORIGINAL", "MODIFIED"])]
    diff: Option<Vec<PathBuf>>,

    /// Render tiles and deploy to this HF Space (e.g. username/my-vis);
    /// bucket is auto-named as <namespace>/<repo>_bucket
    #[arg(long, conflicts_with = "output")]
    space: Option<String>,

    /// Treat inputs as a specific format (currently only "safetensors" is supported).
    /// Auto-detected from .safetensors file extension when omitted.
    #[arg(long, value_name = "FORMAT")]
    format: Option<String>,

    /// Regenerate index.html for an existing tiles directory without re-rendering tiles
    #[arg(long, value_name = "TILES_DIR", conflicts_with_all = ["files", "diff", "output", "tiles", "space", "sort"])]
    regen_html: Option<PathBuf>,

    /// Title shown in the HTML info panel (default: "arbvis" or "arbvis diff")
    #[arg(long, value_name = "TITLE")]
    title: Option<String>,

    /// Color regions by xorb ID for xet-backed files; hue = xorb, intensity = byte.
    #[arg(long)]
    show_xet_xorbs: bool,

    /// Draw thin lines at xet chunk boundaries (can be combined with --show-xet-xorbs).
    #[arg(long)]
    show_xet_chunks: bool,
}

async fn run(args: Args) -> anyhow::Result<()> {
    if let Some(ref tile_dir) = args.regen_html {
        return tiled::regen_html(tile_dir);
    }

    let format_safetensors = args.format.as_deref() == Some("safetensors");
    let xet_vis = args.show_xet_xorbs || args.show_xet_chunks;
    let show_xet_chunks = args.show_xet_chunks;

    if xet_vis && args.sort {
        anyhow::bail!("--show-xet-xorbs / --show-xet-chunks are incompatible with --sort");
    }
    if xet_vis && args.diff.is_some() {
        anyhow::bail!("--show-xet-xorbs / --show-xet-chunks are incompatible with --diff");
    }

    // Intercept hf:// tiles output: resolve inputs as HTTP specs and stream directly.
    // For local tiles output: render to disk then optionally upload.
    // _tiles_tempdir keeps the temp dir alive until upload is done.
    let tiles_hf_out: Option<String> = match &args.tiles {
        Some(p) if p.to_string_lossy().starts_with("hf://") => {
            Some(p.to_string_lossy().into_owned())
        }
        _ => None,
    };
    let (_tiles_tempdir, tiles_arg, tiles_upload) = match args.tiles {
        None => (None, None, None),
        Some(ref p) if p.to_string_lossy().starts_with("hf://") => {
            let td = tempfile::tempdir()?;
            let local = td.path().to_path_buf();
            (Some(td), Some(local), Some(p.to_string_lossy().into_owned()))
        }
        Some(p) => (None, Some(p), None),
    };
    let (_output_tempdir, output_arg, output_upload) = match args.output {
        None => (None, None, None),
        Some(ref p) if p.to_string_lossy().starts_with("hf://") => {
            let td = tempfile::tempdir()?;
            let local = td.path().join("output.png");
            (Some(td), Some(local), Some(p.to_string_lossy().into_owned()))
        }
        Some(p) => (None, Some(p), None),
    };

    if let Some(raw_diff_args) = args.diff {
        let diff_input_strs: Vec<String> = raw_diff_args
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let diff_title = args.title.as_deref().unwrap_or("arbvis diff");
        let orig_str = &diff_input_strs[0];
        let mod_str  = &diff_input_strs[1];

        let (sources, total) = if hf_url::is_repo_level(orig_str)? && hf_url::is_repo_level(mod_str)? {
            // Both are repo-level hf:// URLs: list files over API, diff lazily over HTTP.
            // No model weights are downloaded to disk or held in RAM.
            if args.sort {
                anyhow::bail!("--sort is not supported with repo-level hf:// diff inputs");
            }
            let orig_specs = hf_url::list_repo_as_http_specs(orig_str).await
                .with_context(|| format!("listing files in {orig_str}"))?;
            let mod_specs = hf_url::list_repo_as_http_specs(mod_str).await
                .with_context(|| format!("listing files in {mod_str}"))?;
            data::prepare_diff_sources_from_http(&orig_specs, &mod_specs).await?
        } else {
            // At least one side is a local path or single-file hf:// URL.
            let mut diff_args: Vec<PathBuf> = Vec::with_capacity(raw_diff_args.len());
            for p in raw_diff_args {
                diff_args.push(resolve_input(p).await?);
            }
            data::prepare_diff_sources(&diff_args[0], &diff_args[1], format_safetensors).await?
        };
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        // Stream directly to HF — no tiles written to local disk.
        if let Some(ref hf_out_url) = tiles_hf_out {
            if args.sort { anyhow::bail!("--sort is not supported with hf:// tile output"); }
            let hf_out = hf_url::parse_hf_output(hf_out_url)?;
            let _ = tiled::run_tiles_hf_streaming(sources, total, &hf_out, true, diff_title, &diff_input_strs, false).await?;
            return Ok(());
        }
        if let Some(ref space_id) = args.space {
            if args.sort { anyhow::bail!("--sort is not supported with --space diff output"); }
            let bucket_spec = deploy::create_space_bucket(space_id).await?;
            let html = tiled::run_tiles_hf_streaming(sources, total, &bucket_spec, true, diff_title, &diff_input_strs, false).await?;
            deploy::deploy_space_app(space_id, &bucket_spec.repo_id, html).await?;
            return Ok(());
        }
        if let Some(ref tile_dir) = tiles_arg {
            tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, true, diff_title, &diff_input_strs, false).await?;
            if let Some(ref url) = tiles_upload {
                deploy::upload_dir_to(url, tile_dir).await?;
            }
            return Ok(());
        }
        // single::run_single is sync + rayon. Wrap it in spawn_blocking so the
        // tokio runtime can keep driving any other tasks meanwhile.
        let labels = labels.clone();
        let sources_owned = sources;
        let output_arg_owned = output_arg.clone();
        let sort = args.sort;
        let diff_mode = true;
        tokio::task::spawn_blocking(move || {
            single::run_single(&labels, output_arg_owned, sources_owned, total, sort, diff_mode)
        })
        .await
        .map_err(|e| anyhow::anyhow!("run_single join failure: {e}"))??;
        if let (Some(ref url), Some(ref local)) = (&output_upload, &output_arg) {
            deploy::upload_file_to(url, local).await?;
        }
        return Ok(());
    }

    // Deploy-only shortcut: --space + --tiles with no input files/list means
    // the tiles directory is already fully rendered; just deploy it without
    // re-running the renderer (which would otherwise read empty stdin and
    // overwrite labels.json with a useless "stdin" entry).
    // Only applies when --tiles is a local path (not hf://).
    if args.files.is_empty() && args.file_list.is_none() && tiles_upload.is_none() {
        if let (Some(ref tile_dir), Some(ref space_id)) = (&tiles_arg, &args.space) {
            deploy::run_deploy(tile_dir, space_id).await?;
            return Ok(());
        }
    }

    let mut files = args.files;
    if let Some(list_path) = args.file_list {
        let reader: Box<dyn Read> = if list_path == PathBuf::from("-") {
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
                files.push(PathBuf::from(trimmed));
            }
        }
    }

    // When tiles output is hf://, stream tiles directly to Hub (zero local disk).
    if let Some(ref hf_out_url) = tiles_hf_out {
        if args.sort {
            anyhow::bail!("--sort is not supported with hf:// tile output");
        }
        let input_strs: Vec<String> = files
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let mut specs: Vec<InputSpec> = Vec::with_capacity(files.len());
        for p in &files {
            let s = p.to_string_lossy();
            if s.starts_with("hf://") {
                specs.push(InputSpec::Remote(hf_url::resolve_to_http(p).await?));
            } else {
                specs.push(InputSpec::Local(p.clone()));
            }
        }
        let (mut sources, total) = data::prepare_sources_from_specs(&specs, format_safetensors)?;
        if xet_vis {
            data::populate_xet_terms(&mut sources).await?;
        }
        // Materialize remote sources to local cache — see materialize_http_sources
        // for why per-range hf-hub xet calls are too expensive for the tile workload.
        data::materialize_http_sources(&mut sources).await?;
        let hf_out = hf_url::parse_hf_output(hf_out_url)?;
        let stream_title = args.title.as_deref().unwrap_or("arbvis");
        let _ = tiled::run_tiles_hf_streaming(sources, total, &hf_out, false, stream_title, &input_strs, show_xet_chunks).await?;
        return Ok(());
    }

    let original_inputs: Vec<String> = files
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();

    // With xet visualization, we keep hf:// inputs as remote specs (no
    // download, so the file's xet_hash and per-tile byte ranges are
    // available). Without xet, we download to the local cache as before.
    let tile_title = args.title.as_deref().unwrap_or("arbvis");
    let (sources, total) = if xet_vis {
        let mut specs: Vec<InputSpec> = Vec::new();
        for p in &files {
            let s = p.to_string_lossy();
            if s.starts_with("hf://") {
                if hf_url::is_repo_level(&s)? {
                    let listed = hf_url::list_repo_as_http_specs(&s).await
                        .with_context(|| format!("listing files in {s}"))?;
                    for (_, spec) in listed {
                        specs.push(InputSpec::Remote(spec));
                    }
                } else {
                    specs.push(InputSpec::Remote(hf_url::resolve_to_http(p).await?));
                }
            } else {
                specs.push(InputSpec::Local(p.clone()));
            }
        }
        let (mut sources, total) = data::prepare_sources_from_specs(&specs, format_safetensors)?;
        // Capture xet term metadata while the source is still remote, then
        // materialize each file to local cache. Per-range hf-hub xet calls
        // are too expensive for the tile workload — one whole-file download
        // amortises the xet setup over the entire file (which we read every
        // byte of anyway during render).
        data::populate_xet_terms(&mut sources).await?;
        data::materialize_http_sources(&mut sources).await?;
        (sources, total)
    } else {
        let mut resolved: Vec<PathBuf> = Vec::with_capacity(files.len());
        for p in files {
            resolved.push(resolve_input(p).await?);
        }
        data::prepare_sources(&resolved, format_safetensors)?
    };
    let display_files: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();

    if let Some(ref tile_dir) = tiles_arg {
        tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, false, tile_title, &original_inputs, show_xet_chunks).await?;
        if let Some(ref space_id) = args.space {
            deploy::run_deploy(tile_dir, space_id).await?;
        }
        if let Some(ref url) = tiles_upload {
            deploy::upload_dir_to(url, tile_dir).await?;
        }
        return Ok(());
    }

    if let Some(ref space_id) = args.space {
        if args.sort { anyhow::bail!("--sort is not supported with --space output"); }
        let bucket_spec = deploy::create_space_bucket(space_id).await?;
        let html = tiled::run_tiles_hf_streaming(sources, total, &bucket_spec, false, tile_title, &original_inputs, show_xet_chunks).await?;
        deploy::deploy_space_app(space_id, &bucket_spec.repo_id, html).await?;
        return Ok(());
    }

    // single::run_single is sync + rayon. spawn_blocking keeps the tokio
    // runtime responsive.
    let display_files_owned = display_files.clone();
    let sources_owned = sources;
    let output_arg_owned = output_arg.clone();
    let sort = args.sort;
    tokio::task::spawn_blocking(move || {
        single::run_single(&display_files_owned, output_arg_owned, sources_owned, total, sort, false)
    })
    .await
    .map_err(|e| anyhow::anyhow!("run_single join failure: {e}"))??;
    if let (Some(ref url), Some(ref local)) = (&output_upload, &output_arg) {
        deploy::upload_file_to(url, local).await?;
    }
    Ok(())
}

/// Resolve an input path: download from HF if it starts with `hf://`.
async fn resolve_input(path: PathBuf) -> anyhow::Result<PathBuf> {
    let display = path.display().to_string();
    hf_url::resolve(&path).await.with_context(|| format!("resolving {display}"))
}


#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    run(args).await
}