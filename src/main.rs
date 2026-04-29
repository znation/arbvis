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

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let readers: Vec<(usize, Box<dyn Read>)> = if args.files.is_empty() {
        vec![(0, Box::new(io::stdin()))]
    } else {
        args.files
            .iter()
            .enumerate()
            .map(|(i, p)| -> Result<(usize, Box<dyn Read>), io::Error> {
                Ok((i, Box::new(BufReader::new(File::open(p)?))))
            })
            .collect::<Result<_, io::Error>>()?
    };

    let num_files = readers.len();

    // Read all bytes upfront so we know total size before sizing the canvas.
    let mut all_bytes: Vec<(usize, u8)> = Vec::new();
    for (file_idx, reader) in readers {
        for b in reader.bytes() {
            all_bytes.push((file_idx, b?));
        }
    }

    let total = all_bytes.len().max(1);
    // Smallest k such that (2^k)^2 >= total, capped at 12 (4096×4096, ~50MB RGB8)
    // to stay within GPU max_buffer_binding_size limits (~128MB).
    let mut k = 1u32;
    while (1usize << (2 * k)) < total {
        k += 1;
    }
    let k = k.min(12);
    let side = 1u32 << k;
    let canvas_size = (side * side) as usize;

    // Subsample if there are more bytes than canvas pixels.
    let stride = if total > canvas_size {
        (total + canvas_size - 1) / canvas_size
    } else {
        1
    };
    let sampled: Vec<(usize, u8)> = all_bytes.iter().step_by(stride).take(canvas_size).cloned().collect();

    let mut img = DynamicImage::new_rgb8(side, side);
    let window = create_window("image", Default::default())?;

    // pixel_file[y * side + x] = which file index painted this pixel
    let mut pixel_file: Vec<Option<usize>> = vec![None; canvas_size];

    for (count, (file_idx, v)) in sampled.iter().enumerate() {
        let (x, y): (u32, u32) = h2xy(count as u64, 1);
        img.put_pixel(x, y, byte_to_pixel(*v));
        pixel_file[y as usize * side as usize + x as usize] = Some(*file_idx);

        if count % 10000 == 0 {
            window.set_image("image-001", img.clone())?;
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

    window.set_image("image-001", img)?;
    window.wait_until_destroyed()?;
    Ok(())
}
