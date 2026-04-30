use std::path::PathBuf;

use ab_glyph::{FontRef, PxScale};
use image::Rgb;
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;

/// Deterministic per-cell jitter for label positions.
/// Returns (jx, jy, neg_x, neg_y) where jx,jy ∈ 0..=100.
pub fn jitter(i: u32, j: u32) -> (u32, u32, bool, bool) {
    let h = i.wrapping_mul(2654435761).wrapping_add(j.wrapping_mul(2246822519));
    let jx = (h & 0xFF) % 101;
    let jy = ((h >> 8) & 0xFF) % 101;
    let neg_x = (h >> 16) & 1 == 1;
    let neg_y = (h >> 17) & 1 == 1;
    (jx, jy, neg_x, neg_y)
}

/// Draw a label for `fi`-th file on the canvas inside its bounding box.
/// Attempts TL grid first, then corners, then a coarse scan.
pub fn draw_file_label(
    fi: usize,
    bbox: (u32, u32, u32, u32),
    files: &[PathBuf],
    canvas: &mut image::ImageBuffer<Rgb<u8>, Vec<u8>>,
    pixel_file: &[Option<usize>],
    font: &FontRef<'static>,
    scale: PxScale,
    side: u32,
) {
    let (x0, y0, x1, y1) = bbox;
    let raw = files[fi]
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let label: String = if raw.chars().count() > 40 {
        let truncated: String = raw.chars().take(40).collect();
        format!("{truncated}…")
    } else {
        raw
    };

    let (text_w, text_h) = text_size(scale, font, &label);
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
            font,
            &label,
        );
        true
    };

    // Phase 1: primary TL grid — preserves multi-label behavior on large files.
    let mut placed_any = false;
    let mut j = 0u32;
    loop {
        let base_y = y0 + 20 + 512 * j;
        if base_y + box_h - 1 > y1 {
            break;
        }
        let mut i = 0u32;
        loop {
            let base_x = x0 + 20 + 512 * i;
            if base_x + box_w - 1 > x1 {
                break;
            }
            let (jx, jy, neg_x, neg_y) = jitter(i, j);
            let label_x = if neg_x {
                base_x.saturating_sub(jx)
            } else {
                base_x + jx
            };
            let label_y = if neg_y {
                base_y.saturating_sub(jy)
            } else {
                base_y + jy
            };
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