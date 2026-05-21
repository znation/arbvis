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
use crate::data::{load_source_data, Data, Source, SourceKind};
use crate::geometry::{hilbert_to_xy_u64, sampled_in_range};
use crate::label::draw_file_label;
use crate::layout::arch::ArchLayout;
use crate::layout::render::{plain_element_color, PADDING_RGB};
use crate::layout::{select_layout, Layout, LayoutMode};
use crate::safetensors::{color_for_pos, DiffFill};
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

/// Maximum side length of the single-image output. Mirrors the GPU buffer
/// size cap baked into the legacy Hilbert path (`(1 << 12)² × 3 ≈ 50 MB`).
const SINGLE_MAX_DIM: u32 = 4096;

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
    // Opportunistic sidecar (config.json / index.json) load. Runs inside
    // the ambient tokio runtime via `block_on` because `run_single` is sync
    // and is already inside `spawn_blocking`. For Hilbert mode we skip —
    // the legacy path doesn't consume the metadata.
    let metas = if matches!(layout_mode, LayoutMode::Hilbert) {
        Vec::new()
    } else {
        tokio::runtime::Handle::current().block_on(crate::data::load_meta_for_sources(&sources))
    };
    if let Some(arch_summary) = metas
        .iter()
        .find_map(|m| m.config.as_ref().map(|c| c.summary()))
    {
        log::info!("model config: {arch_summary}");
    }
    let layout = select_layout(&sources, &cumulative_offsets, total, layout_mode, &metas);
    if let Layout::Architectural(arch) = &layout {
        // Arch mode only handles local (mmap'd / owned) data for now. If any
        // source needs an HTTP/Xet/LazyDiff fetch we fall through to the
        // Hilbert path with a warning, since the architectural single-image
        // renderer is synchronous and we don't want to block a tokio worker
        // inside spawn_blocking on per-pixel HTTP calls.
        let all_local = sources.iter().all(|s| {
            matches!(
                s.kind,
                SourceKind::File(_) | SourceKind::Buffered(_) | SourceKind::Diff { .. }
            )
        });
        if all_local && !diff_mode && !show_xet_xorbs {
            return run_single_arch(files, output, &sources, arch);
        }
        log::warn!(
            "architectural single-image layout requires local non-diff non-xet inputs; falling back to hilbert"
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

/// Synchronous architectural-mode single-image render. Renders every tensor
/// into a downsampled overview that fits inside `SINGLE_MAX_DIM²`; preserves
/// the per-tensor 2D aspect via independent integer downsampling.
fn run_single_arch(
    _files: &[PathBuf],
    output: Option<PathBuf>,
    sources: &[Source],
    layout: &ArchLayout,
) -> anyhow::Result<()> {
    let (canvas_w, canvas_h) = (layout.width, layout.height);
    // Global integer downscale so the largest dimension fits in SINGLE_MAX_DIM.
    let max_dim = canvas_w.max(canvas_h).max(1);
    let scale: u32 = max_dim.div_ceil(SINGLE_MAX_DIM).max(1);
    let out_w = (canvas_w / scale).max(1);
    let out_h = (canvas_h / scale).max(1);

    let mut img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image::ImageBuffer::new(out_w, out_h);
    for p in img.pixels_mut() {
        *p = PADDING_RGB;
    }

    // Open each source as `Data` so we can borrow its bytes synchronously
    // via `Deref` (panics for HTTP/Xet/LazyDiff — but `run_single`'s
    // dispatcher already gated those out above).
    let data: Vec<Data> = sources
        .iter()
        .map(load_source_data)
        .collect::<anyhow::Result<Vec<_>>>()?;

    let pixel_lut = build_pixel_lut();

    // Per tensor, sample every `scale`-th element in row-major order and
    // paint it into the output image. The sampling is intentionally simple
    // pixel-skip (not box-averaged) — for a 4096²-bound overview the loss
    // of detail vs box-averaging is invisible at this scale and the loop is
    // a memory-bound walk.
    for t in &layout.tensors {
        let cols = t.tensor_cols;
        let rows = t.tensor_rows;
        let elem = t.dtype.element_size() as u64;
        let stride = cols * elem;
        let src_idx = t.source_idx;
        let src_bytes: &[u8] = match &data[src_idx] {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
            _ => continue,
        };
        // Tensor byte start, local to its source.
        let local_off = t
            .tensor_byte_start
            .saturating_sub(sources[..src_idx].iter().map(|s| s.byte_size).sum::<u64>());

        // Output rect after scaling.
        let out_x0 = t.canvas_x / scale;
        let out_y0 = t.canvas_y / scale;
        let out_x1 = ((t.canvas_x + cols.min(u32::MAX as u64) as u32) / scale).min(out_w);
        let out_y1 = ((t.canvas_y + rows.min(u32::MAX as u64) as u32) / scale).min(out_h);
        if out_x1 <= out_x0 || out_y1 <= out_y0 {
            continue;
        }

        for oy in out_y0..out_y1 {
            // Map output y back to a representative tensor row.
            let dy = (oy - out_y0) as u64 * scale as u64;
            if dy >= rows {
                break;
            }
            let row_off = local_off + dy * stride;
            for ox in out_x0..out_x1 {
                let dx = (ox - out_x0) as u64 * scale as u64;
                if dx >= cols {
                    break;
                }
                let elem_off = (row_off + dx * elem) as usize;
                if elem_off + elem as usize > src_bytes.len() {
                    continue;
                }
                let color = plain_element_color(
                    t.dtype,
                    &src_bytes[elem_off..elem_off + elem as usize],
                    0,
                    &pixel_lut,
                );
                img.put_pixel(ox, oy, color);
            }
        }
    }

    if let Some(path) = output {
        image::DynamicImage::ImageRgb8(img).save(&path)?;
        return Ok(());
    }

    // Interactive window: just show the final image and wait for close.
    let pixels: Vec<u32> = img
        .pixels()
        .map(|p| ((p[0] as u32) << 16) | ((p[1] as u32) << 8) | (p[2] as u32))
        .collect();
    let mut window = Window::new(
        "arbvis (arch layout) — press Esc or close to quit",
        out_w as usize,
        out_h as usize,
        WindowOptions::default(),
    )
    .map_err(|e| anyhow::anyhow!("failed to open preview window: {e}"))?;
    window.set_target_fps(10);
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        window
            .update_with_buffer(&pixels, out_w as usize, out_h as usize)
            .map_err(|e| anyhow::anyhow!("window update error: {e}"))?;
    }
    Ok(())
}

/// Legacy Hilbert-curve single-image renderer.
///
/// `source_data` is parallel to `sources`. Lazy/remote `Data` variants are
/// read per chunk by blocking on `rt`, which lets this otherwise-synchronous
/// rayon pipeline drive `--diff` and other `LazyDiff`-backed sources that
/// can't be `Deref`d.
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
    // Per-pixel color combining (optional) xorb hue, (optional) dtype hue, and
    // byte intensity. When both xorb and dtype are present, the two hues are
    // averaged then modulated by the byte value; either alone falls back to
    // pure xet or pure dtype-byte; with neither, the plain byte LUT.
    let pixel_color = |byte: u8, abs_byte: u64, dtype: Option<Rgb<u8>>| -> Rgb<u8> {
        match (xorb_map.color_idx_at(abs_byte), dtype) {
            (Some(idx), Some(d)) => {
                let t = tableau[idx as usize];
                let s = byte as u32;
                Rgb([
                    (((d[0] as u32 + t[0] as u32) * s + 255) / 510) as u8,
                    (((d[1] as u32 + t[1] as u32) * s + 255) / 510) as u8,
                    (((d[2] as u32 + t[2] as u32) * s + 255) / 510) as u8,
                ])
            }
            (Some(idx), None) => {
                let t = tableau[idx as usize];
                let s = byte as u16;
                Rgb([
                    ((t[0] as u16 * s + 127) / 255) as u8,
                    ((t[1] as u16 * s + 127) / 255) as u8,
                    ((t[2] as u16 * s + 127) / 255) as u8,
                ])
            }
            (None, Some(d)) => {
                let s = byte as u16;
                Rgb([
                    ((d[0] as u16 * s + 127) / 255) as u8,
                    ((d[1] as u16 * s + 127) / 255) as u8,
                    ((d[2] as u16 * s + 127) / 255) as u8,
                ])
            }
            (None, None) => pixel_lut[byte as usize],
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
                let dtype_ranges = sources[src_idx]
                    .safetensors
                    .as_ref()
                    .map(|st| st.color_ranges.as_slice());
                let unmatched_fill: Option<DiffFill> = match &sources[src_idx].kind {
                    SourceKind::UnmatchedRegion { fill } => Some(*fill),
                    _ => None,
                };
                // Bytes are only needed for xet (entropy modulation) and for
                // plain mode (byte LUT). Dtype and unmatched-region paths are
                // position-only.
                let needs_bytes = unmatched_fill.is_none()
                    && (xet_mode || dtype_ranges.is_none());
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
                        } else if !needs_bytes {
                            let ranges = dtype_ranges.unwrap();
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
                                    let dtype = dtype_ranges
                                        .map(|r| color_for_pos(cur_byte - src_global_start, r));
                                    let color = pixel_color(b, cur_byte, dtype);
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
