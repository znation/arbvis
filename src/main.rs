mod color;
mod data;
mod deploy;
mod geometry;
mod hf_upload;
mod hf_url;
mod label;
mod safetensors;
mod single;
mod tiled;

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
}

fn run(args: Args) -> anyhow::Result<()> {
    if let Some(ref tile_dir) = args.regen_html {
        return tiled::regen_html(tile_dir);
    }

    let format_safetensors = args.format.as_deref() == Some("safetensors");

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
        let diff_args: Vec<PathBuf> = raw_diff_args
            .into_iter()
            .map(resolve_input)
            .collect::<anyhow::Result<_>>()?;
        let (sources, total) =
            data::prepare_diff_sources(&diff_args[0], &diff_args[1], format_safetensors)?;
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        if let Some(ref tile_dir) = tiles_arg {
            tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, true)?;
            if let Some(ref space_id) = args.space {
                deploy::run_deploy(tile_dir, space_id)?;
            }
            if let Some(ref url) = tiles_upload {
                hf_url::upload_dir_to(url, tile_dir)?;
            }
            return Ok(());
        }
        if let Some(ref space_id) = args.space {
            let tile_dir = derive_space_tile_dir(space_id);
            tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, true)?;
            deploy::run_deploy(&tile_dir, space_id)?;
            return Ok(());
        }
        single::run_single(&labels, output_arg.clone(), sources, total, args.sort, true)?;
        if let (Some(ref url), Some(ref local)) = (&output_upload, &output_arg) {
            hf_url::upload_file_to(url, local)?;
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
            deploy::run_deploy(tile_dir, space_id)?;
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
        let specs: Vec<InputSpec> = files
            .iter()
            .map(|p| {
                let s = p.to_string_lossy();
                if s.starts_with("hf://") {
                    hf_url::resolve_to_http(p).map(InputSpec::Remote)
                } else {
                    Ok(InputSpec::Local(p.clone()))
                }
            })
            .collect::<anyhow::Result<_>>()?;
        let (sources, total) = data::prepare_sources_from_specs(&specs, format_safetensors)?;
        let hf_out = hf_url::parse_hf_output(hf_out_url)?;
        tiled::run_tiles_hf_streaming(sources, total, &hf_out, false)?;
        return Ok(());
    }

    // Resolve any hf:// paths in the file list (downloading to local cache).
    let files: Vec<PathBuf> = files
        .into_iter()
        .map(resolve_input)
        .collect::<anyhow::Result<_>>()?;

    let (sources, total) = data::prepare_sources(&files, format_safetensors)?;
    let display_files: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();

    if let Some(ref tile_dir) = tiles_arg {
        tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, false)?;
        if let Some(ref space_id) = args.space {
            deploy::run_deploy(tile_dir, space_id)?;
        }
        if let Some(ref url) = tiles_upload {
            hf_url::upload_dir_to(url, tile_dir)?;
        }
        return Ok(());
    }

    if let Some(ref space_id) = args.space {
        let tile_dir = derive_space_tile_dir(space_id);
        tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, false)?;
        deploy::run_deploy(&tile_dir, space_id)?;
        return Ok(());
    }

    single::run_single(&display_files, output_arg.clone(), sources, total, args.sort, false)?;
    if let (Some(ref url), Some(ref local)) = (&output_upload, &output_arg) {
        hf_url::upload_file_to(url, local)?;
    }
    Ok(())
}

fn derive_space_tile_dir(space_id: &str) -> PathBuf {
    let repo = space_id.split('/').last().unwrap_or(space_id);
    PathBuf::from(repo)
}

/// Resolve an input path: download from HF if it starts with `hf://`.
fn resolve_input(path: PathBuf) -> anyhow::Result<PathBuf> {
    hf_url::resolve(&path).with_context(|| format!("resolving {}", path.display()))
}


fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    run(args)
}