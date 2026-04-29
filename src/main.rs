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
    const W: u32 = 400;
    const H: u32 = 400;
    const LEN: usize = (W * H) as usize;
    let mut hilbert_cache: [Option<(u32, u32)>; LEN] = [None; LEN];

    let mut img = DynamicImage::new_rgb8(W, H);
    let window = create_window("image", Default::default())?;

    // pixel_file[y * W + x] = which file index painted this pixel
    let mut pixel_file: Vec<Option<usize>> = vec![None; LEN];

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
    let mut count = 0usize;
    'outer: for (file_idx, reader) in readers {
        for possible_value in reader.bytes() {
            if count >= LEN {
                break 'outer;
            }
            match possible_value {
                Ok(v) => {
                    if hilbert_cache[count].is_none() {
                        hilbert_cache[count] = Some(h2xy(count as u64, 1));
                    }
                    let (x, y) = hilbert_cache[count].unwrap();

                    if x < W && y < H {
                        img.put_pixel(x, y, byte_to_pixel(v));
                        pixel_file[y as usize * W as usize + x as usize] = Some(file_idx);
                    }

                    if count % 1000 == 0 {
                        window.set_image("image-001", img.clone())?;
                    }
                }
                Err(e) => return Err(Box::new(e)),
            }
            count += 1;
        }
    }

    // When multiple files are given, mark pixels on the border between files black.
    // A border pixel is any pixel whose 4-neighbor was painted by a different file.
    if num_files > 1 {
        for y in 0..H {
            for x in 0..W {
                let idx = y as usize * W as usize + x as usize;
                if let Some(file_idx) = pixel_file[idx] {
                    let is_border = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                        .iter()
                        .any(|(dx, dy)| {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && nx < W as i32 && ny >= 0 && ny < H as i32 {
                                let nidx = ny as usize * W as usize + nx as usize;
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
