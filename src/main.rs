use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::PathBuf;

use clap::Parser;
use fast_hilbert::h2xy;
use image::{DynamicImage, GenericImage, Rgba};
use show_image::create_window;

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

    /// Write the canvas to this PNG file instead of displaying a window
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn byte_to_pixel(v: u8) -> Rgba<u8> {
    // color scheme from
    // https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/
    if v == 0 {
        Rgba([0, 0, 0, 255])
    } else if v == 0xFF {
        Rgba([255, 255, 255, 255])
    } else if v <= 0x1F {
        let value = ((v as f32 - 0x01 as f32) / (0x1F as f32 - 0x01 as f32)) * 255.0;
        Rgba([0, value as u8, 0, 255])
    } else if v <= 0x7E {
        let value = ((v as f32 - 0x20 as f32) / (0x7E as f32 - 0x20 as f32)) * 255.0;
        Rgba([0, 0, value as u8, 255])
    } else {
        let value = ((v as f32 - 0x7F as f32) / (0xFE as f32 - 0x7F as f32)) * 255.0;
        Rgba([value as u8, 0, 0, 255])
    }
}

struct Source {
    file_idx: usize,
    reader: Box<dyn Read>,
}

/// Build sources and return total byte count.
/// Files are opened for streaming; stdin is buffered into memory (size unknown until read).
fn prepare_sources(files: &[PathBuf]) -> Result<(Vec<Source>, u64), Box<dyn std::error::Error>> {
    if files.is_empty() {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                reader: Box::new(io::Cursor::new(buf)),
            }],
            len,
        ));
    }

    let mut sources = Vec::with_capacity(files.len());
    let mut total = 0u64;
    for (i, path) in files.iter().enumerate() {
        let byte_count = std::fs::metadata(path)?.len();
        total += byte_count;
        sources.push(Source {
            file_idx: i,
            reader: Box::new(BufReader::new(File::open(path)?)),
        });
    }
    Ok((sources, total))
}

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let num_files = args.files.len().max(1);
    let (sources, total) = prepare_sources(&args.files)?;

    let total_usize = (total as usize).max(1);
    // Smallest k such that (2^k)^2 >= total, capped at 12 (4096×4096, ~50MB RGB8)
    // to stay within GPU max_buffer_binding_size limits (~128MB).
    let mut k = 1u32;
    while (1usize << (2 * k)) < total_usize {
        k += 1;
    }
    let k = k.min(12);
    let side = 1u32 << k;
    let canvas_size = (side * side) as usize;

    // Subsample if there are more bytes than canvas pixels.
    let stride = if total_usize > canvas_size {
        (total_usize + canvas_size - 1) / canvas_size
    } else {
        1
    } as u64;

    let mut img = DynamicImage::new_rgb8(side, side);
    let window = if args.output.is_none() {
        Some(create_window("image", Default::default())?)
    } else {
        None
    };

    // pixel_file[y * side + x] = which file index painted this pixel
    let mut pixel_file: Vec<Option<usize>> = vec![None; canvas_size];
    let mut byte_pos: u64 = 0;
    let mut pixel_count: usize = 0;

    'outer: for source in sources {
        for b in source.reader.bytes() {
            let b = b?;
            if byte_pos % stride == 0 {
                let (x, y): (u32, u32) = h2xy(pixel_count as u64, 1);
                img.put_pixel(x, y, byte_to_pixel(b));
                pixel_file[y as usize * side as usize + x as usize] = Some(source.file_idx);

                if let Some(ref w) = window {
                    if pixel_count % 10000 == 0 {
                        w.set_image("image-001", img.clone())?;
                    }
                }

                pixel_count += 1;
                if pixel_count >= canvas_size {
                    break 'outer;
                }
            }
            byte_pos += 1;
        }
    }

    // When multiple files are given, mark pixels on the border between files black.
    // A border pixel is any pixel whose 4-neighbor was painted by a different file.
    if num_files > 1 {
        for y in 0..side {
            for x in 0..side {
                let idx = y as usize * side as usize + x as usize;
                if let Some(file_idx) = pixel_file[idx] {
                    let is_border = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                        .iter()
                        .any(|(dx, dy)| {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < side as i32 && ny >= 0 && ny < side as i32 {
                                let nidx = ny as usize * side as usize + nx as usize;
                                pixel_file[nidx].map_or(false, |nf| nf != file_idx)
                            } else {
                                false
                            }
                        });
                    if is_border {
                        img.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                    }
                }
            }
        }
    }

    if let Some(path) = args.output {
        img.save(&path)?;
    } else if let Some(w) = window {
        w.set_image("image-001", img)?;
        w.wait_until_destroyed()?;
    }
    Ok(())
}
