use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use ab_glyph::{FontRef, PxScale};
use std::io::IsTerminal;

use image::{DynamicImage, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use show_image::create_window;
use show_image::event::WindowEvent;

use crate::color::build_pixel_lut;
use crate::data::{open_source_data, Source};
use crate::geometry::{sampled_in_range, hilbert_to_xy_u64};
use crate::label::draw_file_label;

/// Render a single Hilbert-curve image (non-tiled mode).
pub fn run_single(
    files: &[PathBuf],
    output: Option<PathBuf>,
    sources: Vec<Source>,
    total: u64,
) -> anyhow::Result<()> {
    let num_files = files.len().max(1);
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

    let pixel_lut = build_pixel_lut();

    let mut img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image::ImageBuffer::new(side, side);

    let window = if output.is_none() {
        Some(create_window("image", Default::default())?)
    } else {
        None
    };

    let cancelled = Arc::new(AtomicBool::new(false));
    let _event_thread = if let Some(ref w) = window {
        let cancelled_c = Arc::clone(&cancelled);
        let rx = w.event_channel()?;
        Some(std::thread::spawn(move || {
            while let Ok(event) = rx.recv() {
                if matches!(event, WindowEvent::CloseRequested(_) | WindowEvent::Destroyed(_)) {
                    cancelled_c.store(true, Ordering::Release);
                    break;
                }
            }
            cancelled_c.store(true, Ordering::Release);
        }))
    } else {
        None
    };

    let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf"))
        .expect("bundled DejaVuSans.ttf is valid");
    let scale = PxScale { x: 14.0, y: 14.0 };

    // pixel_file[y * side + x] = which file index painted this pixel
    let mut pixel_file: Vec<Option<usize>> = vec![None; canvas_size];
    let mut bboxes: Vec<Option<(u32, u32, u32, u32)>> = vec![None; num_files];

    let pb = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    // Open all sources for random access.
    let source_data: Vec<_> = sources
        .iter()
        .map(open_source_data)
        .collect::<anyhow::Result<_>>()?;

    // Split each source into chunks for fine-grained parallelism.
    // Targeting ~4 MB per chunk so every core gets work even for single-file input.
    let chunk_bytes = (4 * 1024 * 1024u64).max(stride);
    let mut chunks: Vec<(usize, usize, u64, u64, u64, u64)> = Vec::new();
    {
        let mut b = 0u64;
        let mut p = 0u64;
        for (src_idx, s) in sources.iter().enumerate() {
            let fi = s.file_idx;
            let src_end = b + s.byte_size;
            let mut cb = b;
            let mut cp = p;
            while cb < src_end {
                let ce = (cb + chunk_bytes).min(src_end);
                chunks.push((fi, src_idx, b, cb, ce, cp));
                cp += sampled_in_range(cb, ce, stride);
                cb = ce;
            }
            p += sampled_in_range(b, src_end, stride);
            b += s.byte_size;
        }
    }

    // Background display thread for interactive mode.
    let stop_display = Arc::new(AtomicBool::new(false));
    let display_thread = if let Some(ref w) = window {
        let img_ptr = img.as_ptr() as usize;
        let stop = Arc::clone(&stop_display);
        let cancelled_disp = Arc::clone(&cancelled);
        let w_c = w.clone();
        let side_c = side;
        Some(std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) && !cancelled_disp.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(100));
                let buf: Vec<u8> = unsafe {
                    std::slice::from_raw_parts(
                        img_ptr as *const u8,
                        side_c as usize * side_c as usize * 3,
                    )
                }
                .to_vec();
                if let Some(ib) =
                    image::ImageBuffer::<Rgb<u8>, _>::from_raw(side_c, side_c, buf)
                {
                    let _ = w_c.set_image("image-001", DynamicImage::ImageRgb8(ib));
                }
            }
        }))
    } else {
        None
    };

    // Parallel rendering: each chunk writes exclusively to non-overlapping
    // pixel ranges, so concurrent writes to `img` and `pixel_file` are race-free.
    let img_base = img.as_mut_ptr() as usize;
    let pf_base = pixel_file.as_mut_ptr() as usize;
    let pb_shared: Option<Arc<ProgressBar>> = pb.map(Arc::new);
    let canvas_u = canvas_size as u64;
    let cancelled_proc = Arc::clone(&cancelled);

    let chunk_results: Vec<(usize, Option<(u32, u32, u32, u32)>)> = chunks
        .par_iter()
        .map(|&(fi, src_idx, src_global_start, chunk_b_start, chunk_b_end, chunk_pixel_start)| {
            if chunk_pixel_start >= canvas_u || cancelled_proc.load(Ordering::Acquire) {
                return (fi, None);
            }
            let data = &source_data[src_idx];
            let local_start = (chunk_b_start - src_global_start) as usize;
            let local_end = (chunk_b_end - src_global_start) as usize;
            let bytes = &data[local_start..local_end];

            let mut cur_byte = chunk_b_start;
            let mut cur_pixel = chunk_pixel_start as usize;
            let mut bbox: Option<(u32, u32, u32, u32)> = None;

            for &b in bytes {
                if cur_byte % stride == 0 {
                    let (x, y) = hilbert_to_xy_u64(cur_pixel as u64, k as u8);
                    let color = pixel_lut[b as usize];
                    let pixel_idx = y as usize * side as usize + x as usize;
                    unsafe {
                        let p = (img_base as *mut u8).add(pixel_idx * 3);
                        p.write(color[0]);
                        p.add(1).write(color[1]);
                        p.add(2).write(color[2]);
                        (pf_base as *mut Option<usize>).add(pixel_idx).write(Some(fi));
                    }
                    bbox = Some(match bbox {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => {
                            (x0.min(x), y0.min(y), x1.max(x), y1.max(y))
                        }
                    });
                    cur_pixel += 1;
                    if cur_pixel >= canvas_size {
                        break;
                    }
                }
                cur_byte += 1;
            }
            if let Some(ref pb) = pb_shared {
                pb.inc(chunk_b_end - chunk_b_start);
            }
            (fi, bbox)
        })
        .collect();

    // Merge per-chunk bboxes into per-file bboxes.
    for (fi, bbox) in chunk_results {
        if let Some(b) = bbox {
            bboxes[fi] = Some(match bboxes[fi] {
                None => b,
                Some((x0, y0, x1, y1)) => {
                    (x0.min(b.0), y0.min(b.1), x1.max(b.2), y1.max(b.3))
                }
            });
        }
    }

    // Stop background display thread before mutating img further.
    stop_display.store(true, Ordering::Release);
    if let Some(t) = display_thread {
        let _ = t.join();
    }

    if cancelled.load(Ordering::Acquire) {
        return Ok(());
    }

    if let Some(ref pb) = pb_shared {
        pb.finish_and_clear();
    }

    // When multiple files are given, mark border pixels black.
    if num_files > 1 {
        let img_ptr = img.as_mut_ptr() as usize;
        let pf = &pixel_file;
        let side_u = side as usize;
        (0..side as usize).into_par_iter().for_each(|y| {
            for x in 0..side_u {
                let idx = y * side_u + x;
                if let Some(file_idx) = pf[idx] {
                    let is_border = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                        .iter()
                        .any(|(dx, dy)| {
                            let nx = x as i32 + *dx;
                            let ny = y as i32 + *dy;
                            if nx >= 0 && nx < side_u as i32 && ny >= 0 && ny < side_u as i32 {
                                let nidx = ny as usize * side_u + nx as usize;
                                pf[nidx].map_or(false, |nf| nf != file_idx)
                            } else {
                                false
                            }
                        });
                    if is_border {
                        unsafe {
                            let p = (img_ptr as *mut u8).add(idx * 3);
                            p.write(0);
                            p.add(1).write(0);
                            p.add(2).write(0);
                        }
                    }
                }
            }
        });
    }

    if let Some(path) = output {
        if !files.is_empty() {
            for (fi, _) in files.iter().enumerate() {
                if let Some(bbox) = bboxes[fi] {
                    draw_file_label(fi, bbox, files, &mut img, &pixel_file, &font, scale, side);
                }
            }
        }
        DynamicImage::ImageRgb8(img).save(&path)?;
    } else if let Some(w) = window {
        if !files.is_empty() {
            for (fi, _) in files.iter().enumerate() {
                if let Some(bbox) = bboxes[fi] {
                    draw_file_label(fi, bbox, files, &mut img, &pixel_file, &font, scale, side);
                }
            }
        }
        w.set_image("image-001", DynamicImage::ImageRgb8(img))?;
        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }
        w.wait_until_destroyed()?;
    }

    Ok(())
}