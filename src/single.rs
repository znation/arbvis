use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ab_glyph::{FontRef, PxScale};
use image::{DynamicImage, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use minifb::{Window, WindowOptions};
use rayon::prelude::*;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Data, DiffFill, Source, SourceKind};
use crate::geometry::{hilbert_to_xy_u64, sampled_in_range};
use crate::label::draw_file_label;
use crate::layout::{select_layout, LayoutMode};
use crate::xet::{XorbMap, TABLEAU_20};

/// Two-pixel-wide diagonal crosshatch on (x, y) for unmatched-region sources.
/// Kept identical in spirit to the tile renderer's pattern so the single-image
/// output is visually consistent with the tiled viewer.
#[inline]
fn is_crosshatch_stripe_single(x: u32, y: u32) -> bool {
    const PERIOD: u32 = 8;
    const STRIPE: u32 = 2;
    let a = (x + y) % PERIOD;
    let b = (x + (PERIOD - y % PERIOD)) % PERIOD;
    a < STRIPE || b < STRIPE
}

/// Render a single image (non-tiled mode). Chooses between the legacy global-
/// Hilbert layout and the architectural layout based on `layout_mode`.
pub fn run_single(
    files: &[PathBuf],
    output: Option<PathBuf>,
    sources: Vec<Source>,
    total: u64,
    diff_mode: bool,
    show_xet_xorbs: bool,
    layout_mode: LayoutMode,
    registry: &crate::registry::Registry,
) -> anyhow::Result<()> {
    // Compute cumulative source offsets for layout selection.
    let cumulative_offsets: Vec<u64> = {
        let mut v = Vec::with_capacity(sources.len());
        let mut o = 0u64;
        for s in &sources {
            v.push(o);
            o += s.byte_size;
        }
        v
    };
    let layout = select_layout(
        &sources,
        &cumulative_offsets,
        total,
        layout_mode,
        diff_mode,
        registry,
    );
    // A structure-aware layout may ship a single-image renderer, keyed by its
    // layout id (the single-image analog of the tiled loader/renderer pair).
    // Use it when it can draw this invocation; otherwise fall back to the
    // byte-Hilbert path below. (The arch renderer, for one, is synchronous and
    // only handles local non-diff non-xet sources — that gate lives in its
    // `applicable`, not here.)
    if let Some(renderer) = registry.single_renderers.get(layout.id()) {
        if renderer.applicable(&sources, diff_mode, show_xet_xorbs) {
            return renderer.render(files, output, &sources, layout.as_ref());
        }
        log::warn!(
            "single-image renderer for layout `{}` can't draw these inputs; \
             falling back to hilbert",
            layout.id()
        );
    }
    // Open each source as a `Data` handle (sync — mmap / lightweight clone)
    // before dispatching, so the rayon workers can be handed ready-to-use
    // borrows. The captured tokio `Handle` lets those workers drive
    // `Data::fetch_range` for `Http` / `Xet` / `LazyDiff` sources (which
    // panic on `Deref`) — that's what makes `--diff --output` work for any
    // LazyDiff source, including JSON.
    let source_data: Vec<Data> = sources
        .iter()
        .map(load_source_data)
        .collect::<anyhow::Result<_>>()?;
    let rt = tokio::runtime::Handle::current();
    run_single_hilbert(
        files,
        output,
        sources,
        source_data,
        total,
        diff_mode,
        show_xet_xorbs,
        rt,
    )
}

fn run_single_hilbert(
    files: &[PathBuf],
    output: Option<PathBuf>,
    sources: Vec<Source>,
    source_data: Vec<Data>,
    total: u64,
    diff_mode: bool,
    show_xet_xorbs: bool,
    rt: tokio::runtime::Handle,
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
        total_usize.div_ceil(canvas_size)
    } else {
        1
    } as u64;

    let pixel_lut = if diff_mode {
        build_diff_signed_lut()
    } else {
        build_pixel_lut()
    };

    // Build xorb map only when xorb coloring was explicitly requested.
    let xorb_map = if show_xet_xorbs {
        XorbMap::build(sources.iter().scan(0u64, |off, s| {
            let cur = *off;
            *off += s.byte_size;
            Some((s.xet_terms.as_deref(), cur))
        }))
    } else {
        XorbMap {
            global_ranges: Vec::new(),
        }
    };
    let tableau: [Rgb<u8>; 20] = {
        let mut arr = [Rgb([0u8, 0, 0]); 20];
        for (i, c) in TABLEAU_20.iter().enumerate() {
            arr[i] = Rgb(*c);
        }
        arr
    };

    let img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image::ImageBuffer::new(side, side);

    let font = FontRef::try_from_slice(include_bytes!("DejaVuSans.ttf"))
        .expect("bundled DejaVuSans.ttf is valid");
    let scale = PxScale { x: 14.0, y: 14.0 };

    // pixel_file[y * side + x] = which file index painted this pixel
    let pixel_file: Vec<Option<usize>> = vec![None; canvas_size];
    let bboxes_init: Vec<Option<(u32, u32, u32, u32)>> = vec![None; num_files];

    // Both paths track the same amount of work (one pass over total bytes).
    // Added to the global multi so log lines interleave cleanly (and so the
    // hidden draw target kicks in automatically for non-TTY runs).
    let pb = {
        let pb = crate::progress::multi().add(ProgressBar::new(total));
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} {msg} ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_message("source bytes rendered");
        Some(pb)
    };

    // Compute per-source byte and pixel offsets.
    let chunk_bytes = (4 * 1024 * 1024u64).max(stride);
    let mut chunks_by_source: Vec<Vec<(usize, u64, u64, u64, u64)>> = vec![vec![]; sources.len()];
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
                chunks_by_source[src_idx].push((fi, b, cb, ce, cp));
                cp += sampled_in_range(cb, ce, stride);
                cb = ce;
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
            &sources,
            &source_data,
            &chunks_by_source,
            &cancelled,
            img_base,
            pf_base,
            canvas_u,
            canvas_size,
            side,
            k,
            stride,
            &pixel_lut,
            &xorb_map,
            &tableau,
            &pb_shared,
            &rt,
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
            pb.finish_and_clear();
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
                &sources,
                &source_data,
                &chunks_by_source,
                &cancelled_bg,
                img_ptr,
                pf_ptr,
                canvas_u,
                canvas_size,
                side,
                k,
                stride,
                &pixel_lut,
                &xorb_map,
                &tableau,
                &pb_shared,
                &rt,
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
                        Some((x0, y0, x1, y1)) => {
                            (x0.min(b.0), y0.min(b.1), x1.max(b.2), y1.max(b.3))
                        }
                    });
                }
            }
            if let Some(ref pb) = pb_shared {
                pb.finish_and_clear();
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
            let is_open = window.is_open() && !window.is_key_down(minifb::Key::Escape);

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
            window
                .update_with_buffer(&pixels, side as usize, side as usize)
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
        let final_pixels: Vec<u32> = img
            .pixels()
            .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32))
            .collect();
        window
            .update_with_buffer(&final_pixels, side as usize, side as usize)
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
    source_data: &[Data],
    chunks_by_source: &[Vec<(usize, u64, u64, u64, u64)>],
    cancelled: &AtomicBool,
    img_base: usize,
    pf_base: usize,
    canvas_u: u64,
    canvas_size: usize,
    side: u32,
    k: u32,
    stride: u64,
    pixel_lut: &[Rgb<u8>; 256],
    xorb_map: &XorbMap,
    tableau: &[Rgb<u8>; 20],
    pb_shared: &Option<Arc<ProgressBar>>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Vec<(usize, Option<(u32, u32, u32, u32)>)>> {
    let xet_mode = !xorb_map.is_empty();
    // Per-pixel color combining the (optional) xorb hue and byte intensity.
    // With a xorb present, the tableau hue is modulated by the byte value;
    // otherwise the plain byte LUT.
    let pixel_color = |byte: u8, abs_byte: u64| -> Rgb<u8> {
        match xorb_map.color_idx_at(abs_byte) {
            Some(idx) => {
                let t = tableau[idx as usize];
                let s = byte as u16;
                Rgb([
                    ((t[0] as u16 * s + 127) / 255) as u8,
                    ((t[1] as u16 * s + 127) / 255) as u8,
                    ((t[2] as u16 * s + 127) / 255) as u8,
                ])
            }
            None => pixel_lut[byte as usize],
        }
    };
    let chunk_results: Vec<(usize, Option<(u32, u32, u32, u32)>)> = chunks_by_source
        .par_iter()
        .enumerate()
        .map(
            |(src_idx, source_chunks)| -> anyhow::Result<Vec<(usize, Option<(u32, u32, u32, u32)>)>> {
                if source_chunks.is_empty() {
                    return Ok(vec![]);
                }
                let unmatched_fill: Option<DiffFill> = match &sources[src_idx].kind {
                    SourceKind::UnmatchedRegion { fill } => Some(*fill),
                    _ => None,
                };
                // arbvis byte-only no longer has a dtype-only path; the
                // only no-bytes case left is UnmatchedRegion. Kept as a
                // local for parity with the original shape — fed into the
                // chunk loop below.
                let _needs_bytes = unmatched_fill.is_none() && xet_mode;
                let data: &Data = &source_data[src_idx];
                let results: anyhow::Result<Vec<(usize, Option<(u32, u32, u32, u32)>)>> = source_chunks
                    .par_iter()
                    .map(|&(fi, src_global_start, chunk_b_start, chunk_b_end, chunk_pixel_start)| -> anyhow::Result<(usize, Option<(u32, u32, u32, u32)>)> {
                        if chunk_pixel_start >= canvas_u || cancelled.load(Ordering::Acquire) {
                            return Ok((fi, None));
                        }

                        let mut cur_pixel = chunk_pixel_start as usize;
                        let mut bbox: Option<(u32, u32, u32, u32)> = None;

                        if let Some(fill) = unmatched_fill {
                            // No bytes to read — paint the crosshatch pattern
                            // for every strided pixel in this chunk.
                            let (stripe_c, base_c) = fill.colors();
                            let first = chunk_b_start
                                + (stride - chunk_b_start % stride) % stride;
                            for _strided_b in (first..chunk_b_end).step_by(stride as usize) {
                                if cur_pixel >= canvas_size {
                                    break;
                                }
                                let (x, y) = hilbert_to_xy_u64(cur_pixel as u64, k as u8);
                                let color = if is_crosshatch_stripe_single(x, y) {
                                    stripe_c
                                } else {
                                    base_c
                                };
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
                            let local_start = (chunk_b_start - src_global_start) as usize;
                            let local_end = (chunk_b_end - src_global_start) as usize;
                            let chunk_len = local_end - local_start;
                            // Local-backed Data (`Mapped` / `Owned` /
                            // `ZeroFill`) is Deref'd zero-copy; remote/lazy
                            // Data (`Http`, `Xet`, `LazyDiff`) is fetched per
                            // chunk via `block_on` so this sync rayon worker
                            // can drive the async fetcher without panicking
                            // on Deref.
                            let fetched: Option<Vec<u8>> = if data.is_local() {
                                None
                            } else {
                                Some(rt.block_on(
                                    data.fetch_range(local_start as u64, chunk_len),
                                )?)
                            };
                            let bytes: &[u8] = match &fetched {
                                Some(v) => v,
                                None => &data[local_start..local_end],
                            };
                            for (i, &b) in bytes.iter().enumerate() {
                                let cur_byte = chunk_b_start + i as u64;
                                if cur_byte.is_multiple_of(stride) {
                                    let (x, y) = hilbert_to_xy_u64(cur_pixel as u64, k as u8);
                                    let color = pixel_color(b, cur_byte);
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
                            }
                        }
                        if let Some(ref pb) = pb_shared {
                            pb.inc(chunk_b_end - chunk_b_start);
                        }
                        Ok((fi, bbox))
                    })
                    .collect();
                results
            },
        )
        .collect::<anyhow::Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    // When multiple files are given, mark border pixels black.
    if sources
        .iter()
        .map(|s| s.file_idx)
        .collect::<std::collections::HashSet<_>>()
        .len()
        > 1
    {
        let side_u = side as usize;
        // Reconstruct pixel_file reference from raw pointer for border pass
        let pf_slice: &[Option<usize>] =
            unsafe { std::slice::from_raw_parts(pf_base as *const Option<usize>, canvas_size) };
        (0..side_u).into_par_iter().for_each(|y| {
            for x in 0..side_u {
                let idx = y * side_u + x;
                if let Some(file_idx) = pf_slice[idx] {
                    let is_border =
                        [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)]
                            .iter()
                            .any(|(dx, dy)| {
                                let nx = x as i32 + *dx;
                                let ny = y as i32 + *dy;
                                if nx >= 0 && nx < side_u as i32 && ny >= 0 && ny < side_u as i32 {
                                    let nidx = ny as usize * side_u + nx as usize;
                                    pf_slice[nidx].is_some_and(|nf| nf != file_idx)
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
