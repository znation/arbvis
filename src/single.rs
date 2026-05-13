use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ab_glyph::{FontRef, PxScale};
use std::io::IsTerminal;

use image::{DynamicImage, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use minifb::{Window, WindowOptions};
use rayon::prelude::*;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Histogram, Source};
use crate::geometry::{sampled_in_range, hilbert_to_xy_u64};
use crate::label::draw_file_label;
use crate::safetensors::color_for_pos;

/// Render a single Hilbert-curve image (non-tiled mode).
pub fn run_single(
    files: &[PathBuf],
    output: Option<PathBuf>,
    sources: Vec<Source>,
    total: u64,
    sort: bool,
    diff_mode: bool,
) -> anyhow::Result<()> {
    if sort && sources.iter().any(|s| s.safetensors.is_some()) {
        anyhow::bail!(
            "--sort is incompatible with safetensors files: sort reorders bytes by value, \
             which destroys positional dtype information"
        );
    }

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

    let pixel_lut = if diff_mode { build_diff_signed_lut() } else { build_pixel_lut() };

    let img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image::ImageBuffer::new(side, side);

    let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf"))
        .expect("bundled DejaVuSans.ttf is valid");
    let scale = PxScale { x: 14.0, y: 14.0 };

    // pixel_file[y * side + x] = which file index painted this pixel
    let pixel_file: Vec<Option<usize>> = vec![None; canvas_size];
    let bboxes_init: Vec<Option<(u32, u32, u32, u32)>> = vec![None; num_files];

    // Both paths track the same amount of work (one pass over total bytes).
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

    // Compute per-source byte and pixel offsets.
    let chunk_bytes = (4 * 1024 * 1024u64).max(stride);
    let mut src_byte_starts: Vec<u64> = Vec::with_capacity(sources.len());
    let mut src_pixel_starts: Vec<u64> = Vec::with_capacity(sources.len());
    let mut chunks_by_source: Vec<Vec<(usize, u64, u64, u64, u64)>> =
        if sort { vec![] } else { vec![vec![]; sources.len()] };
    {
        let mut b = 0u64;
        let mut p = 0u64;
        for (src_idx, s) in sources.iter().enumerate() {
            src_byte_starts.push(b);
            src_pixel_starts.push(p);
            if !sort {
                let fi = s.file_idx;
                let src_end = b + s.byte_size;
                let mut cb = b;
                let mut cp = p;
                while cb < src_end {
                    let ce = (cb + chunk_bytes).min(src_end);
                    chunks_by_source[src_idx].push((fi, b, cb, ce, cp));
                    cp += sampled_in_range(cb, ce, stride);
                    cb = ce;
                }
            }
            p += sampled_in_range(b, b + s.byte_size, stride);
            b += s.byte_size;
        }
    }

    if output.is_some() {
        // ─── File output path: render directly ───────────────────────────────
        let mut img = img;
        let mut pixel_file = pixel_file;
        let mut bboxes = bboxes_init;

        let cancelled = Arc::new(AtomicBool::new(false));
        let img_base = img.as_mut_ptr() as usize;
        let pf_base = pixel_file.as_mut_ptr() as usize;
        let pb_shared: Option<Arc<ProgressBar>> = pb.map(|pb| {
            pb.enable_steady_tick(Duration::from_millis(100));
            Arc::new(pb)
        });
        let canvas_u = canvas_size as u64;

        let chunk_results = render_chunks(
            &sources, &chunks_by_source, sort, &src_byte_starts, &src_pixel_starts,
            &cancelled, img_base, pf_base, canvas_u, canvas_size, side, k, stride,
            &pixel_lut, &pb_shared,
        )?;

        for (fi, bbox) in chunk_results {
            if let Some(b) = bbox {
                bboxes[fi] = Some(match bboxes[fi] {
                    None => b,
                    Some((x0, y0, x1, y1)) => (x0.min(b.0), y0.min(b.1), x1.max(b.2), y1.max(b.3)),
                });
            }
        }

        if let Some(ref pb) = pb_shared {
            pb.finish();
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
        }

    } else {
        // ─── Interactive path: rendering in background, minifb on main thread ─
        let cancelled = Arc::new(AtomicBool::new(false));
        let render_done = Arc::new(AtomicBool::new(false));
        let bboxes_shared: Arc<Mutex<Vec<Option<(u32, u32, u32, u32)>>>> =
            Arc::new(Mutex::new(bboxes_init));

        // Box the image and pixel_file so we can safely capture pointer addresses.
        let mut img = Box::new(img);
        let mut pixel_file = Box::new(pixel_file);
        let img_ptr = img.as_mut_ptr() as usize;
        let pf_ptr = pixel_file.as_mut_ptr() as usize;

        let cancelled_bg = Arc::clone(&cancelled);
        let done_bg = Arc::clone(&render_done);
        let bboxes_bg = Arc::clone(&bboxes_shared);
        let pb_shared: Option<Arc<ProgressBar>> = pb.map(|pb| {
            pb.enable_steady_tick(Duration::from_millis(100));
            Arc::new(pb)
        });
        let canvas_u = canvas_size as u64;

        let bg_result: Arc<Mutex<Option<anyhow::Result<()>>>> = Arc::new(Mutex::new(None));
        let bg_result_store = Arc::clone(&bg_result);

        let bg_thread = std::thread::spawn(move || {
            let result = render_chunks(
                &sources, &chunks_by_source, sort, &src_byte_starts, &src_pixel_starts,
                &cancelled_bg, img_ptr, pf_ptr, canvas_u, canvas_size, side, k, stride,
                &pixel_lut, &pb_shared,
            );

            let chunk_results = match result {
                Ok(r) => r,
                Err(e) => {
                    *bg_result_store.lock().unwrap() = Some(Err(e));
                    done_bg.store(true, Ordering::Release);
                    return;
                }
            };

            let mut bboxes = bboxes_bg.lock().unwrap();
            for (fi, bbox) in chunk_results {
                if let Some(b) = bbox {
                    bboxes[fi] = Some(match bboxes[fi] {
                        None => b,
                        Some((x0, y0, x1, y1)) => (x0.min(b.0), y0.min(b.1), x1.max(b.2), y1.max(b.3)),
                    });
                }
            }
            if let Some(ref pb) = pb_shared {
                pb.finish();
            }

            done_bg.store(true, Ordering::Release);
        });

        // minifb event loop on main thread.
        let mut window = Window::new(
            "arbvis — press Esc or close to quit",
            side as usize,
            side as usize,
            WindowOptions::default(),
        )
        .map_err(|e| anyhow::anyhow!("failed to open preview window: {e}"))?;
        window.set_target_fps(10);

        loop {
            let is_open = window.is_open()
                && !window.is_key_down(minifb::Key::Escape);

            if !is_open {
                cancelled.store(true, Ordering::Release);
                break;
            }

            // Copy Rgb pixels to 0x00RRGGBB u32 buffer (data race with render
            // workers is intentional — stale pixels are acceptable in a live preview).
            let pixels: Vec<u32> = unsafe {
                let ptr = img_ptr as *const u8;
                let n = side as usize * side as usize;
                (0..n)
                    .map(|i| {
                        let r = *ptr.add(i * 3) as u32;
                        let g = *ptr.add(i * 3 + 1) as u32;
                        let b = *ptr.add(i * 3 + 2) as u32;
                        (r << 16) | (g << 8) | b
                    })
                    .collect()
            };
            window.update_with_buffer(&pixels, side as usize, side as usize)
                .map_err(|e| anyhow::anyhow!("window update error: {e}"))?;

            if render_done.load(Ordering::Acquire) {
                break;
            }
        }

        bg_thread.join().expect("render thread panicked");

        // Check for render errors.
        if let Some(err) = bg_result.lock().unwrap().take() {
            return err;
        }

        if cancelled.load(Ordering::Acquire) {
            return Ok(());
        }

        // Draw labels and show final image.
        let mut img = *img;
        let pixel_file = *pixel_file;
        let bboxes = bboxes_shared.lock().unwrap();

        if !files.is_empty() {
            for (fi, _) in files.iter().enumerate() {
                if let Some(bbox) = bboxes[fi] {
                    draw_file_label(fi, bbox, files, &mut img, &pixel_file, &font, scale, side);
                }
            }
        }
        let final_pixels: Vec<u32> = img.pixels().map(|p| {
            ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32)
        }).collect();
        window.update_with_buffer(&final_pixels, side as usize, side as usize)
            .map_err(|e| anyhow::anyhow!("window update error: {e}"))?;

        // Keep window open until user closes it.
        while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
            window.update();
        }
    }

    Ok(())
}

/// Run the parallel rendering across all sources, returning per-file bounding boxes.
///
/// Both interactive and file-output paths share this logic.
#[allow(clippy::too_many_arguments)]
fn render_chunks(
    sources: &[Source],
    chunks_by_source: &[Vec<(usize, u64, u64, u64, u64)>],
    sort: bool,
    src_byte_starts: &[u64],
    src_pixel_starts: &[u64],
    cancelled: &AtomicBool,
    img_base: usize,
    pf_base: usize,
    canvas_u: u64,
    canvas_size: usize,
    side: u32,
    k: u32,
    stride: u64,
    pixel_lut: &[Rgb<u8>; 256],
    pb_shared: &Option<Arc<ProgressBar>>,
) -> anyhow::Result<Vec<(usize, Option<(u32, u32, u32, u32)>)>> {
    let chunk_results: Vec<(usize, Option<(u32, u32, u32, u32)>)> = if sort {
        (0..sources.len())
            .into_par_iter()
            .map(|src_idx| -> anyhow::Result<(usize, Option<(u32, u32, u32, u32)>)> {
                let source = &sources[src_idx];
                let hist = Histogram::build(source, pb_shared.as_deref())?;
                let prefix = hist.prefix_sums();
                let fi = source.file_idx;
                let src_byte_start = src_byte_starts[src_idx];
                let mut cur_pixel = src_pixel_starts[src_idx];
                let mut bbox: Option<(u32, u32, u32, u32)> = None;

                for v in 0usize..=255 {
                    if cancelled.load(Ordering::Acquire) {
                        break;
                    }
                    let n = hist.0[v];
                    if n == 0 {
                        continue;
                    }
                    let global_byte_start = src_byte_start + prefix[v];
                    let global_byte_end = src_byte_start + prefix[v + 1];
                    let n_pixels = sampled_in_range(global_byte_start, global_byte_end, stride);
                    let color = pixel_lut[v];

                    for p_off in 0..n_pixels {
                        let p = cur_pixel + p_off;
                        if p >= canvas_u {
                            break;
                        }
                        let (x, y) = hilbert_to_xy_u64(p, k as u8);
                        let pixel_idx = y as usize * side as usize + x as usize;
                        unsafe {
                            let ptr = (img_base as *mut u8).add(pixel_idx * 3);
                            ptr.write(color[0]);
                            ptr.add(1).write(color[1]);
                            ptr.add(2).write(color[2]);
                            (pf_base as *mut Option<usize>).add(pixel_idx).write(Some(fi));
                        }
                        bbox = Some(match bbox {
                            None => (x, y, x, y),
                            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                        });
                    }
                    cur_pixel += n_pixels;
                }

                Ok((fi, bbox))
            })
            .collect::<anyhow::Result<_>>()?
    } else {
        chunks_by_source
            .par_iter()
            .enumerate()
            .map(
                |(src_idx, source_chunks)| -> anyhow::Result<Vec<(usize, Option<(u32, u32, u32, u32)>)>> {
                    if source_chunks.is_empty() {
                        return Ok(vec![]);
                    }
                    let dtype_ranges = sources[src_idx]
                        .safetensors
                        .as_ref()
                        .map(|st| st.color_ranges.as_slice());
                    let data = if dtype_ranges.is_none() {
                        Some(load_source_data(&sources[src_idx])?)
                    } else {
                        None
                    };
                    let results = source_chunks
                        .par_iter()
                        .map(|&(fi, src_global_start, chunk_b_start, chunk_b_end, chunk_pixel_start)| {
                            if chunk_pixel_start >= canvas_u || cancelled.load(Ordering::Acquire) {
                                return (fi, None);
                            }

                            let mut cur_pixel = chunk_pixel_start as usize;
                            let mut bbox: Option<(u32, u32, u32, u32)> = None;

                            if let Some(ranges) = dtype_ranges {
                                let first = chunk_b_start
                                    + (stride - chunk_b_start % stride) % stride;
                                for strided_b in (first..chunk_b_end).step_by(stride as usize) {
                                    if cur_pixel >= canvas_size {
                                        break;
                                    }
                                    let (x, y) = hilbert_to_xy_u64(cur_pixel as u64, k as u8);
                                    let color = color_for_pos(strided_b - src_global_start, ranges);
                                    let pixel_idx = y as usize * side as usize + x as usize;
                                    unsafe {
                                        let p = (img_base as *mut u8).add(pixel_idx * 3);
                                        p.write(color[0]);
                                        p.add(1).write(color[1]);
                                        p.add(2).write(color[2]);
                                        (pf_base as *mut Option<usize>)
                                            .add(pixel_idx)
                                            .write(Some(fi));
                                    }
                                    bbox = Some(match bbox {
                                        None => (x, y, x, y),
                                        Some((x0, y0, x1, y1)) => {
                                            (x0.min(x), y0.min(y), x1.max(x), y1.max(y))
                                        }
                                    });
                                    cur_pixel += 1;
                                }
                            } else {
                                let data = data.as_ref().unwrap();
                                let local_start = (chunk_b_start - src_global_start) as usize;
                                let local_end = (chunk_b_end - src_global_start) as usize;
                                let bytes = &data[local_start..local_end];
                                let mut cur_byte = chunk_b_start;

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
                                            (pf_base as *mut Option<usize>)
                                                .add(pixel_idx)
                                                .write(Some(fi));
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
                            }
                            if let Some(ref pb) = pb_shared {
                                pb.inc(chunk_b_end - chunk_b_start);
                            }
                            (fi, bbox)
                        })
                        .collect();
                    Ok(results)
                },
            )
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect()
    };

    // When multiple files are given, mark border pixels black.
    if sources.iter().map(|s| s.file_idx).collect::<std::collections::HashSet<_>>().len() > 1 {
        let side_u = side as usize;
        // Reconstruct pixel_file reference from raw pointer for border pass
        let pf_slice: &[Option<usize>] = unsafe {
            std::slice::from_raw_parts(pf_base as *const Option<usize>, canvas_size)
        };
        (0..side_u).into_par_iter().for_each(|y| {
            for x in 0..side_u {
                let idx = y * side_u + x;
                if let Some(file_idx) = pf_slice[idx] {
                    let is_border = [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                        .iter()
                        .any(|(dx, dy)| {
                            let nx = x as i32 + *dx;
                            let ny = y as i32 + *dy;
                            if nx >= 0 && nx < side_u as i32 && ny >= 0 && ny < side_u as i32 {
                                let nidx = ny as usize * side_u + nx as usize;
                                pf_slice[nidx].map_or(false, |nf| nf != file_idx)
                            } else {
                                false
                            }
                        });
                    if is_border {
                        unsafe {
                            let p = (img_base as *mut u8).add(idx * 3);
                            p.write(0);
                            p.add(1).write(0);
                            p.add(2).write(0);
                        }
                    }
                }
            }
        });
    }

    Ok(chunk_results)
}
