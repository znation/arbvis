mod color;
mod data;
mod geometry;
mod label;
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
    files: Vec<PathBuf>,

    /// Read file list from this file (one path per line), or - for stdin
    #[arg(short = 'l', long)]
    file_list: Option<PathBuf>,

    /// Write the canvas to this PNG file instead of displaying a window
    #[arg(short, long, conflicts_with = "tiles")]
    output: Option<PathBuf>,

    /// Write a tiled pyramid to this directory for Leaflet.js viewing
    #[arg(short, long, conflicts_with = "output")]
    tiles: Option<PathBuf>,
}

fn run(args: Args) -> anyhow::Result<()> {
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

    let (sources, total) = data::prepare_sources(&files)?;

    if let Some(tile_dir) = args.tiles {
        return tiled::run_tiles(sources, total, tile_dir);
    }

    single::run_single(&files, args.output, sources, total)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let has_output = args.output.is_some() || args.tiles.is_some();

    if has_output {
        run(args)
    } else {
        show_image::run_context(move || run(args));
    }
}