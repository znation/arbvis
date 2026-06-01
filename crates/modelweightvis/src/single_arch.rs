//! Architectural single-image renderer. Lifted from the pre-split
//! arbvis::single::run_single_arch — the byte-only fork of single.rs
//! drops the arch branch and delegates to this module via
//! [`arbvis::SingleImageArchHook`] (registered as
//! [`crate::hooks::ArchSingleImageHook`]).
//!
//! The arch single-image is a downsampled overview that fits inside a
//! `SINGLE_MAX_DIM`-bounded canvas. Each tensor's natural row-major
//! element grid is integer-stride-sampled into its on-canvas
//! footprint; the byte-Hilbert path would map every byte to a curve
//! position, but for arch we already have a 2D placement and want to
//! preserve it.

use std::path::PathBuf;

use arbvis::color::build_pixel_lut;
use arbvis::{load_source_data, Data, LayoutShape, Source};
use image::Rgb;
use minifb::{Window, WindowOptions};

use crate::layout::arch::ArchLayout;
use crate::layout::render::{plain_element_color, PADDING_RGB};

/// Cap for the single-image arch canvas's longest side. The full
/// `ArchLayout` canvas can run into the hundreds of millions of pixels;
/// for the non-tiled single-image we want something user-viewable in a
/// minifb window or as a PNG, so we integer-downscale by the smallest
/// factor that brings `max(width, height)` under this cap.
const SINGLE_MAX_DIM: u32 = 4096;

/// Concrete impl behind [`crate::hooks::ArchSingleImageHook`]. arbvis's
/// `single::run_single` calls this when the selected layout's id is
/// `"arch"`, all sources are local, and not in diff/xet mode.
///
/// Walks each placed tensor in `layout`, integer-stride-samples its
/// natural element grid into the on-canvas display footprint, and
/// paints the sampled pixels via the dtype-aware
/// [`plain_element_color`] (`u8 → Rgb` for fixed dtypes, dequant-then-
/// LUT for k-quants). Padding pixels (gaps between tensors / under-
/// sized rows) stay [`PADDING_RGB`]-filled.
///
/// The output ImageBuffer matches what the tiled pipeline would
/// produce at the same downscale ratio — modulo subpixel quantisation,
/// which intentionally drops here (we step elements; the tiler
/// box-averages).  That means the smoke baselines for single-image
/// arch are NOT byte-identical to the tiled pipeline's level-0 pyramid;
/// they're independent contracts.
pub fn run_single_arch(
    _files: &[PathBuf],
    output: Option<PathBuf>,
    sources: &[Source],
    layout: &dyn LayoutShape,
) -> anyhow::Result<()> {
    let arch: &ArchLayout = layout
        .as_any()
        .downcast_ref::<ArchLayout>()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "ArchSingleImageHook invoked with a non-ArchLayout `LayoutShape` \
             (id={:?}); arbvis should have gated on layout.id() == \"arch\"",
                layout.id()
            )
        })?;

    let (canvas_w, canvas_h) = (arch.width, arch.height);
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
    for t in &arch.tensors {
        let cols = t.tensor_cols;
        let rows = t.tensor_rows;
        // On-canvas footprint (element grid scaled by the per-tensor display
        // scale); output pixels map back to elements through this footprint.
        let disp_w = t.disp_w as u64;
        let disp_h = t.disp_h as u64;
        let src_idx = t.source_idx;
        let src_bytes: &[u8] = match &data[src_idx] {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
            _ => continue,
        };
        // Tensor byte start, local to its source. The tensor's element data runs
        // contiguously from here; `plain_element_color` indexes it by *element*
        // index, decoding the dtype's natural stride (fixed / block-quantised) —
        // so we pass the whole tail slice and a flat element index rather than a
        // hand-computed byte offset, which would be wrong for block/packed dtypes.
        let local_off = t
            .tensor_byte_start
            .saturating_sub(sources[..src_idx].iter().map(|s| s.byte_size).sum::<u64>())
            as usize;
        if local_off >= src_bytes.len() {
            continue;
        }
        let tensor_bytes = &src_bytes[local_off..];

        // Output rect after scaling, in terms of the display footprint.
        let out_x0 = t.canvas_x / scale;
        let out_y0 = t.canvas_y / scale;
        let out_x1 = ((t.canvas_x + disp_w.min(u32::MAX as u64) as u32) / scale).min(out_w);
        let out_y1 = ((t.canvas_y + disp_h.min(u32::MAX as u64) as u32) / scale).min(out_h);
        if out_x1 <= out_x0 || out_y1 <= out_y0 {
            continue;
        }

        for oy in out_y0..out_y1 {
            // Output y → display-pixel offset within the footprint → element row.
            let disp_y = (oy - out_y0) as u64 * scale as u64;
            if disp_y >= disp_h {
                break;
            }
            let er = disp_y * rows / disp_h.max(1);
            for ox in out_x0..out_x1 {
                let disp_x = (ox - out_x0) as u64 * scale as u64;
                if disp_x >= disp_w {
                    break;
                }
                let ec = disp_x * cols / disp_w.max(1);
                let flat = (er * cols + ec) as usize;
                let color = plain_element_color(t.dtype, tensor_bytes, flat, &pixel_lut);
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
