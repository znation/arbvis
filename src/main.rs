use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::PathBuf;

use ab_glyph::{FontRef, PxScale};
use clap::Parser;
use fast_hilbert::h2xy;
use image::{DynamicImage, GenericImage, Rgb, Rgba};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
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

    // Draw filename labels in each file's pixel region.
    if !args.files.is_empty() {
        // Compute per-file bounding boxes from pixel_file.
        let mut bboxes: Vec<Option<(u32, u32, u32, u32)>> = vec![None; num_files];
        for y in 0..side {
            for x in 0..side {
                if let Some(fi) = pixel_file[y as usize * side as usize + x as usize] {
                    bboxes[fi] = Some(match bboxes[fi] {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    });
                }
            }
        }

        let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf"))
            .expect("bundled DejaVuSans.ttf is valid");
        let scale = PxScale { x: 14.0, y: 14.0 };
        let canvas = img.as_mut_rgb8().expect("canvas is RGB8");

        for (fi, path) in args.files.iter().enumerate() {
            let Some((x0, y0, x1, y1)) = bboxes[fi] else {
                continue;
            };
            let raw = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let label: String = if raw.chars().count() > 40 {
                let truncated: String = raw.chars().take(40).collect();
                format!("{truncated}…")
            } else {
                raw
            };

            let (text_w, text_h) = text_size(scale, &font, &label);
            let box_w = text_w + 8;
            let box_h = text_h + 8;

            let mut try_place = |label_x: u32, label_y: u32| -> bool {
                if label_x < x0 || label_y < y0 {
                    return false;
                }
                if label_x + box_w - 1 > x1 || label_y + box_h - 1 > y1 {
                    return false;
                }
                let owned = (label_y..label_y + box_h).all(|py| {
                    (label_x..label_x + box_w).all(|px| {
                        pixel_file[py as usize * side as usize + px as usize] == Some(fi)
                    })
                });
                if !owned {
                    return false;
                }
                draw_filled_rect_mut(
                    canvas,
                    Rect::at(label_x as i32, label_y as i32).of_size(box_w, box_h),
                    Rgb([0u8, 0, 0]),
                );
                draw_text_mut(
                    canvas,
                    Rgb([0u8, 255, 0]),
                    (label_x + 4) as i32,
                    (label_y + 4) as i32,
                    scale,
                    &font,
                    &label,
                );
                true
            };

            // Phase 1: primary TL grid — preserves multi-label behavior on large files.
            let mut placed_any = false;
            let mut j = 0u32;
            loop {
                let label_y = y0 + 20 + 512 * j;
                if label_y + box_h - 1 > y1 {
                    break;
                }
                let mut i = 0u32;
                loop {
                    let label_x = x0 + 20 + 512 * i;
                    if label_x + box_w - 1 > x1 {
                        break;
                    }
                    if try_place(label_x, label_y) {
                        placed_any = true;
                    }
                    i += 1;
                }
                j += 1;
            }

            // Phase 2: try the other three corners with the same 20px inset.
            if !placed_any {
                let right_x = (x1 + 1).saturating_sub(box_w + 20);
                let bottom_y = (y1 + 1).saturating_sub(box_h + 20);
                for (cx, cy) in [
                    (right_x, y0 + 20),  // TR
                    (x0 + 20, bottom_y), // BL
                    (right_x, bottom_y), // BR
                ] {
                    if try_place(cx, cy) {
                        placed_any = true;
                        break;
                    }
                }
            }

            // Phase 3: coarse scan of the entire bbox for any owned position.
            if !placed_any {
                const STRIDE: u32 = 8;
                let max_x = (x1 + 1).saturating_sub(box_w);
                let max_y = (y1 + 1).saturating_sub(box_h);
                'scan: for cy in (y0..=max_y).step_by(STRIDE as usize) {
                    for cx in (x0..=max_x).step_by(STRIDE as usize) {
                        if try_place(cx, cy) {
                            break 'scan;
                        }
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
