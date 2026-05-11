mod color;
mod data;
mod deploy;
mod geometry;
mod label;
mod safetensors;
mod single;
mod tiled;

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

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
}

fn run(args: Args) -> anyhow::Result<()> {
    let format_safetensors = args.format.as_deref() == Some("safetensors");

    if let Some(diff_args) = args.diff {
        let (sources, total) =
            data::prepare_diff_sources(&diff_args[0], &diff_args[1], format_safetensors)?;
        let labels: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();
        if let Some(tile_dir) = args.tiles {
            tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, true)?;
            if let Some(ref space_id) = args.space {
                deploy::run_deploy(&tile_dir, space_id)?;
            }
            return Ok(());
        }
        if let Some(ref space_id) = args.space {
            let tile_dir = derive_space_tile_dir(space_id);
            tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, true)?;
            deploy::run_deploy(&tile_dir, space_id)?;
            return Ok(());
        }
        return single::run_single(&labels, args.output, sources, total, args.sort, true);
    }

    // Deploy-only shortcut: --space + --tiles with no input files/list means
    // the tiles directory is already fully rendered; just deploy it without
    // re-running the renderer (which would otherwise read empty stdin and
    // overwrite labels.json with a useless "stdin" entry).
    if args.files.is_empty() && args.file_list.is_none() {
        if let (Some(ref tile_dir), Some(ref space_id)) = (args.tiles.as_ref(), args.space.as_ref()) {
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

    let (sources, total) = data::prepare_sources(&files, format_safetensors)?;
    let display_files: Vec<PathBuf> = sources.iter().map(|s| PathBuf::from(s.name())).collect();

    if let Some(tile_dir) = args.tiles {
        tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, false)?;
        if let Some(ref space_id) = args.space {
            deploy::run_deploy(&tile_dir, space_id)?;
        }
        return Ok(());
    }

    if let Some(ref space_id) = args.space {
        let tile_dir = derive_space_tile_dir(space_id);
        tiled::run_tiles(sources, total, tile_dir.clone(), args.sort, false)?;
        deploy::run_deploy(&tile_dir, space_id)?;
        return Ok(());
    }

    single::run_single(&display_files, args.output, sources, total, args.sort, false)
}

fn derive_space_tile_dir(space_id: &str) -> PathBuf {
    let repo = space_id.split('/').last().unwrap_or(space_id);
    PathBuf::from(repo)
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    let has_output = args.output.is_some() || args.tiles.is_some() || args.space.is_some();

    if has_output {
        run(args)
    } else {
        show_image::run_context(move || run(args));
    }
}