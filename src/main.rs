use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;

use ab_glyph::{FontRef, PxScale};
use clap::Parser;
use fast_hilbert::{h2xy, xy2h};
use image::{DynamicImage, Rgb};
use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use show_image::create_window;
use show_image::event::WindowEvent;

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

    /// Read file list from this file (one path per line), or - for stdin
    #[arg(short = 'l', long)]
    file_list: Option<PathBuf>,

    /// Write the canvas to this PNG file instead of displaying a window
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Write a tiled pyramid to this directory for Leaflet.js viewing
    #[arg(short, long)]
    tiles: Option<PathBuf>,
}

/// Mmapped or in-memory backing for a source's bytes.
enum Data {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Data {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Data::Mapped(m) => m,
            Data::Owned(v) => v,
        }
    }
}

fn byte_to_pixel(v: u8) -> Rgb<u8> {
    // color scheme from
    // https://stairwell.com/resources/hilbert-curves-visualizing-binary-files-with-color-and-patterns/
    if v == 0 {
        Rgb([0, 0, 0])
    } else if v == 0xFF {
        Rgb([255, 255, 255])
    } else if v <= 0x1F {
        let value = ((v as f32 - 0x01 as f32) / (0x1F as f32 - 0x01 as f32)) * 255.0;
        Rgb([0, value as u8, 0])
    } else if v <= 0x7E {
        let value = ((v as f32 - 0x20 as f32) / (0x7E as f32 - 0x20 as f32)) * 255.0;
        Rgb([0, 0, value as u8])
    } else {
        let value = ((v as f32 - 0x7F as f32) / (0xFE as f32 - 0x7F as f32)) * 255.0;
        Rgb([value as u8, 0, 0])
    }
}

enum SourceKind {
    Buffered(Vec<u8>),
    File(PathBuf),
}

struct Source {
    file_idx: usize,
    kind: SourceKind,
    byte_size: u64,
}


/// Build sources and return total byte count.
/// Files are opened lazily (one at a time) to avoid exhausting OS fd limits.
/// Stdin is buffered into memory upfront since its size is unknown.
fn prepare_sources(files: &[PathBuf]) -> Result<(Vec<Source>, u64), Box<dyn std::error::Error>> {
    if files.is_empty() {
        let mut buf = Vec::new();
        io::stdin().read_to_end(&mut buf)?;
        let len = buf.len() as u64;
        return Ok((
            vec![Source {
                file_idx: 0,
                kind: SourceKind::Buffered(buf),
                byte_size: len,
            }],
            len,
        ));
    }

    let mut sources = Vec::with_capacity(files.len());
    let mut total = 0u64;
    for (i, path) in files.iter().enumerate() {
        let size = match std::fs::metadata(path) {
            Ok(m) => m.len(),
            Err(e) => {
                eprintln!("warning: {}: {} — skipping", path.display(), e);
                continue;
            }
        };
        total += size;
        sources.push(Source {
            file_idx: i,
            kind: SourceKind::File(path.clone()),
            byte_size: size,
        });
    }
    Ok((sources, total))
}

/// Count multiples of `stride` in the byte range `[byte_start, byte_end)`.
fn sampled_in_range(byte_start: u64, byte_end: u64, stride: u64) -> u64 {
    if byte_end <= byte_start { return 0; }
    if stride == 1 { return byte_end - byte_start; }
    if byte_start == 0 {
        (byte_end - 1) / stride + 1
    } else {
        (byte_end - 1) / stride - (byte_start - 1) / stride
    }
}

fn draw_file_label(
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

    // Deterministic per-cell jitter: hash (i, j) to ±100px so repeated labels
    // don't all align in a visible grid when zoomed out.
    let jitter = |i: u32, j: u32| -> (u32, u32, bool, bool) {
        let h = (i.wrapping_mul(2654435761)).wrapping_add(j.wrapping_mul(2246822519));
        let jx = (h & 0xFF) % 101; // 0..=100
        let jy = ((h >> 8) & 0xFF) % 101;
        let neg_x = (h >> 16) & 1 == 1;
        let neg_y = (h >> 17) & 1 == 1;
        (jx, jy, neg_x, neg_y)
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
            let label_x = if neg_x { base_x.saturating_sub(jx) } else { base_x + jx };
            let label_y = if neg_y { base_y.saturating_sub(jy) } else { base_y + jy };
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

// ── Tiled / pyramidal output for Leaflet.js ────────────────────────────────

/// Render one 256×256 leaf tile and write it to `tile_path`.
///
/// Each tile at zoom `max_zoom` covers a 256×256-pixel region that corresponds
/// to a contiguous Hilbert sub-curve of exactly 65536 bytes.  Because tiles are
/// independent, many can be rendered simultaneously.
fn render_leaf_tile(
    tile_path: &Path,
    tx: u32,
    ty: u32,
    kh: u8,
    _height: u32,
    height_tiles: u32,
    square_pixels: u64,
    total: u64,
    data: &[Data],
    cumulative: &[u64],
    pixel_lut: &[Rgb<u8>; 256],
) -> Result<(), String> {
    const TILE: u32 = 256;
    let mut img = image::ImageBuffer::<Rgb<u8>, Vec<u8>>::new(TILE, TILE);
    // Which Hilbert square (column block) does this tile fall in?
    let sq = (tx / height_tiles) as u64;
    let sq_off = sq * square_pixels;
    // Local tile coordinates within the square.
    let local_tx = tx % height_tiles;
    for py in 0..TILE {
        let ly = ty * TILE + py;
        for px in 0..TILE {
            let lx = local_tx * TILE + px;
            let local_idx: u64 = xy2h::<u32>(lx, ly, kh);
            let pixel_idx = sq_off + local_idx;
            let color = if pixel_idx < total {
                let src = cumulative.partition_point(|&c| c <= pixel_idx) - 1;
                let byte = data[src][(pixel_idx - cumulative[src]) as usize];
                pixel_lut[byte as usize]
            } else {
                Rgb([0u8, 0, 0])
            };
            img.put_pixel(px, py, color);
        }
    }
    if let Some(parent) = tile_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    img.save(tile_path).map_err(|e| format!("{}: {}", tile_path.display(), e))?;
    Ok(())
}

fn build_pyramid(
    tiles_dir: &Path,
    tile_size: u32,
    max_zoom: u32,
    width_tiles: u32,
    height_tiles: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if max_zoom == 0 {
        return Ok(());
    }
    let half = tile_size / 2;
    for z in (0..max_zoom).rev() {
        let child_z = z + 1;
        let levels_from_max = max_zoom - z;
        let parent_nx = (width_tiles >> levels_from_max).max(1) as usize;
        let parent_ny = (height_tiles >> levels_from_max).max(1) as usize;
        let errs: Vec<String> = (0..parent_ny * parent_nx)
            .into_par_iter()
            .filter_map(|i| {
                let y = i / parent_nx;
                let x = i % parent_nx;
                let result = (|| -> Result<(), String> {
                    let mut parent: image::ImageBuffer<Rgb<u8>, Vec<u8>> =
                        image::ImageBuffer::new(tile_size, tile_size);
                    let mut has_data = false;
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            let cx = 2 * x + dx;
                            let cy = 2 * y + dy;
                            let child_path =
                                tiles_dir.join(format!("{child_z}/{cx}/{cy}.png"));
                            if !child_path.exists() {
                                continue;
                            }
                            let child = image::open(&child_path)
                                .map_err(|e| e.to_string())?
                                .to_rgb8();
                            has_data = true;
                            let raw = child.as_raw();
                            let row_stride = child.width() as usize * 3;
                            for py in 0..half as usize {
                                for px in 0..half as usize {
                                    let mut r = 0u32;
                                    let mut g = 0u32;
                                    let mut b = 0u32;
                                    let mut count = 0u32;
                                    for sy in 0..2usize {
                                        for sx in 0..2usize {
                                            let sx_ = px * 2 + sx;
                                            let sy_ = py * 2 + sy;
                                            if sx_ < child.width() as usize
                                                && sy_ < child.height() as usize
                                            {
                                                let off = sy_ * row_stride + sx_ * 3;
                                                r += raw[off] as u32;
                                                g += raw[off + 1] as u32;
                                                b += raw[off + 2] as u32;
                                                count += 1;
                                            }
                                        }
                                    }
                                    if count > 0 {
                                        let out_x = dx as u32 * half + px as u32;
                                        let out_y = dy as u32 * half + py as u32;
                                        parent.put_pixel(
                                            out_x,
                                            out_y,
                                            Rgb([
                                                (r / count) as u8,
                                                (g / count) as u8,
                                                (b / count) as u8,
                                            ]),
                                        );
                                    }
                                }
                            }
                        }
                    }
                    if has_data {
                        let parent_path = tiles_dir.join(format!("{z}/{x}/{y}.png"));
                        if let Some(p) = parent_path.parent() {
                            std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
                        }
                        parent
                            .save(&parent_path)
                            .map_err(|e| format!("{}: {}", parent_path.display(), e))?;
                    }
                    Ok(())
                })();
                result.err()
            })
            .collect();
        if let Some(e) = errs.into_iter().next() {
            return Err(e.into());
        }
    }
    Ok(())
}

struct FileEntity {
    name: String,
    pixel_x: u32,
    pixel_y: u32,
    hue: u16,
    segments: Vec<(u32, u32, u32, u32)>, // (x0, y0, x1, y1) in pixel-boundary coords
}

// Hilbert sub-quadrant state table.
// CHILD_TABLE[state][i] = (dx, dy, child_state) where (dx,dy) ∈ {0,1}²
// give the child quadrant's position within the parent (units of child_side).
// Derived from fast_hilbert's order-1 LUT.
const CHILD_TABLE: [[(u32, u32, u8); 4]; 4] = [
    [(0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 2)], // state 0
    [(0, 0, 0), (1, 0, 1), (1, 1, 1), (0, 1, 3)], // state 1
    [(1, 1, 3), (0, 1, 2), (0, 0, 2), (1, 0, 0)], // state 2
    [(1, 1, 2), (1, 0, 3), (0, 0, 3), (0, 1, 1)], // state 3
];

fn name_hue(name: &str) -> u16 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    (h.finish() % 360) as u16
}

/// Recursively decompose Hilbert local range [a, b) at `level` (order-`level`
/// curve on a 2^level × 2^level square) into axis-aligned pixel rectangles.
fn decompose_hilbert(
    a: u64, b: u64,
    level: u8,
    x0: u32, y0: u32,
    side: u32,
    state: u8,
    out: &mut Vec<(u32, u32, u32, u32)>,
) {
    let total = (side as u64) * (side as u64);
    if a == 0 && b == total {
        out.push((x0, y0, x0 + side, y0 + side));
        return;
    }
    if level == 0 {
        out.push((x0, y0, x0 + 1, y0 + 1));
        return;
    }
    let child_side = side >> 1;
    let q_size = (child_side as u64) * (child_side as u64);
    for i in 0u64..4 {
        let ca = i * q_size;
        let cb = ca + q_size;
        if a >= cb || b <= ca {
            continue;
        }
        let (dx, dy, child_state) = CHILD_TABLE[state as usize][i as usize];
        decompose_hilbert(
            a.saturating_sub(ca),
            b.min(cb) - ca,
            level - 1,
            x0 + dx * child_side,
            y0 + dy * child_side,
            child_side,
            child_state,
            out,
        );
    }
}

/// Compute the XOR-merged set of a list of intervals: ranges covered by an odd
/// number of input intervals. Used for extracting outer boundary edges.
fn xor_intervals(intervals: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut events: Vec<(u32, i8)> = Vec::with_capacity(intervals.len() * 2);
    for &(lo, hi) in intervals {
        if lo < hi {
            events.push((lo, 1));
            events.push((hi, -1));
        }
    }
    events.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut result = Vec::new();
    let mut count: i32 = 0;
    let mut seg_start = 0u32;
    for (y, delta) in &events {
        let was_odd = count % 2 != 0;
        count += *delta as i32;
        let is_odd = count % 2 != 0;
        if !was_odd && is_odd {
            seg_start = *y;
        } else if was_odd && !is_odd && seg_start < *y {
            result.push((seg_start, *y));
        }
    }
    result
}

/// Compute outer boundary segments of a set of axis-aligned pixel rectangles
/// that exactly tile a region. Returns (x0,y0,x1,y1) segments in pixel-boundary
/// coords: a rect [px, px+w) × [py, py+h) has edges at x=px, x=px+w, y=py, y=py+h.
fn outer_segments(rects: &[(u32, u32, u32, u32)]) -> Vec<(u32, u32, u32, u32)> {
    let mut vert: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut horiz: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    for &(x0, y0, x1, y1) in rects {
        vert.entry(x0).or_default().push((y0, y1));
        vert.entry(x1).or_default().push((y0, y1));
        horiz.entry(y0).or_default().push((x0, x1));
        horiz.entry(y1).or_default().push((x0, x1));
    }
    let mut result = Vec::new();
    for (&x, intervals) in &vert {
        for (lo, hi) in xor_intervals(intervals) {
            result.push((x, lo, x, hi));
        }
    }
    for (&y, intervals) in &horiz {
        for (lo, hi) in xor_intervals(intervals) {
            result.push((lo, y, hi, y));
        }
    }
    result
}

/// Collect all dyadic pixel rectangles for a file's byte range.
fn file_rects(
    byte_start: u64,
    byte_end: u64,
    total_pixels: u64,
    square_pixels: u64,
    num_squares: u32,
    height: u32,
    kh: u8,
) -> Vec<(u32, u32, u32, u32)> {
    let byte_end = byte_end.min(total_pixels);
    if byte_end <= byte_start {
        return Vec::new();
    }
    let mut all_rects = Vec::new();
    for sq in 0..num_squares as u64 {
        let sq_start = sq * square_pixels;
        let sq_end = sq_start + square_pixels;
        let local_a = byte_start.max(sq_start).saturating_sub(sq_start);
        let local_b = byte_end.min(sq_end).saturating_sub(sq_start);
        if local_b <= local_a {
            continue;
        }
        let x_off = sq as u32 * height;
        decompose_hilbert(local_a, local_b, kh, x_off, 0, height, 0, &mut all_rects);
    }
    all_rects
}

/// Area-weighted centroid of a set of axis-aligned pixel rectangles.
/// Returns None if the rect set is empty.
fn rects_centroid(rects: &[(u32, u32, u32, u32)]) -> Option<(u32, u32)> {
    let mut total_area = 0f64;
    let mut wx = 0f64;
    let mut wy = 0f64;
    for &(x0, y0, x1, y1) in rects {
        let area = (x1 - x0) as f64 * (y1 - y0) as f64;
        wx += area * (x0 + x1) as f64 / 2.0;
        wy += area * (y0 + y1) as f64 / 2.0;
        total_area += area;
    }
    if total_area == 0.0 {
        return None;
    }
    Some(((wx / total_area) as u32, (wy / total_area) as u32))
}

fn write_leaflet_html(
    dir: &Path,
    world_w: u32,
    max_zoom: u32,
    height: u32,
    entities: &[FileEntity],
) -> Result<(), Box<dyn std::error::Error>> {
    // L.CRS.Simple uses transformation (1,0,-1,0): pixel.y = -lat.
    // World height is always 256 units (one tile at zoom 0 in y).
    // World width = 256 * 2^(kw-kh), covering extra columns for wide rectangles.
    // Pixel (x,y) -> Leaflet [lat, lng]: lat = -y*256/height, lng = x*256/height
    let entities_json: String = {
        let entries: Vec<String> = entities
            .iter()
            .map(|e| {
                let escaped = e.name.replace('\\', "\\\\").replace('"', "\\\"");
                let segs: Vec<String> = e
                    .segments
                    .iter()
                    .map(|&(x0, y0, x1, y1)| format!("[{},{},{},{}]", x0, y0, x1, y1))
                    .collect();
                format!(
                    "{{\"name\":\"{}\",\"x\":{},\"y\":{},\"hue\":{},\"segs\":[{}]}}",
                    escaped,
                    e.pixel_x,
                    e.pixel_y,
                    e.hue,
                    segs.join(",")
                )
            })
            .collect();
        format!("[{}]", entries.join(","))
    };

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>arbvis tiled</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .file-label {{
      background: rgba(0,0,0,0.65);
      color: #fff;
      padding: 2px 5px;
      font: 11px/1.4 monospace;
      white-space: nowrap;
      border-radius: 2px;
      pointer-events: none;
      transform: translate(-50%, -50%);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{
      crs: L.CRS.Simple,
      minZoom: 0,
      maxZoom: {max_zoom},
      preferCanvas: true,
    }});
    L.tileLayer('tiles/{{z}}/{{x}}/{{y}}.png', {{
      tileSize: 256,
      bounds: [[-256, 0], [0, {world_w}]],
      noWrap: true,
      attribution: 'arbvis'
    }}).addTo(map);
    map.fitBounds([[-256, 0], [0, {world_w}]]);

    var HEIGHT = {height};
    var labels = {entities_json};
    labels.forEach(function(l) {{
      if (!l.segs || l.segs.length === 0) return;
      var ll = l.segs.map(function(s) {{
        return [
          [-(s[1] / HEIGHT) * 256, (s[0] / HEIGHT) * 256],
          [-(s[3] / HEIGHT) * 256, (s[2] / HEIGHT) * 256],
        ];
      }});
      L.polyline(ll, {{
        color: 'hsl(' + l.hue + ',70%,60%)',
        weight: 1,
        opacity: 0.9,
        fill: false,
        interactive: false,
      }}).addTo(map);
    }});
    labels.forEach(function(l) {{
      var lat = -(l.y / HEIGHT) * 256;
      var lng =  (l.x / HEIGHT) * 256;
      L.marker([lat, lng], {{
        icon: L.divIcon({{
          className: 'file-label',
          html: l.name,
          iconSize: [0, 0],
          iconAnchor: [0, 0]
        }}),
        interactive: false
      }}).addTo(map);
    }});
  </script>
</body>
</html>"#,
        max_zoom = max_zoom,
        world_w = world_w,
        height = height,
        entities_json = entities_json,
    );
    std::fs::write(dir.join("index.html"), html)?;
    Ok(())
}

fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    // Find s = ceil(log2(total)), minimum 16 so the image is at least 256×256.
    // Split into kh = floor(s/2) (height) and kw = ceil(s/2) (width).
    // When s is odd this gives a 2:1 rectangle, halving wasted space vs a square.
    let mut s = 16u32;
    while (1u64 << s) < total {
        s += 1;
    }
    let kh = s / 2;        // height exponent: height = 2^kh pixels
    let kw = (s + 1) / 2;  // width exponent:  width  = 2^kw pixels
    let height = 1u32 << kh;
    let width = 1u32 << kw;
    let tile_size = 256u32;
    let max_zoom = kh - 8;  // kh >= 8 guaranteed by minimum s=16
    let width_tiles = width / tile_size;
    let height_tiles = height / tile_size;
    // World width for Leaflet: height = 256 units, width = 256 * 2^(kw-kh)
    let world_w = 256u32 << (kw - kh);
    // Each Hilbert square covers height×height pixels; tiles are laid left-to-right.
    let square_pixels: u64 = (height as u64) * (height as u64);
    let total_pixels: u64 = width as u64 * height as u64;
    let num_squares = 1u32 << (kw - kh);

    let pixel_lut: [Rgb<u8>; 256] = {
        let mut lut = [Rgb([0u8, 0, 0]); 256];
        for v in 0u16..=255 {
            lut[v as usize] = byte_to_pixel(v as u8);
        }
        lut
    };

    // Build cumulative byte-start offsets so any pixel_idx can be resolved to
    // (source_index, local_byte_offset) in O(log n) via binary search.
    let mut cumulative_offsets: Vec<u64> = Vec::with_capacity(sources.len());
    {
        let mut off = 0u64;
        for s in &sources {
            cumulative_offsets.push(off);
            off += s.byte_size;
        }
    }

    // Memory-map each file (or keep stdin data in-memory) for random access.
    let data: Vec<Data> = sources
        .iter()
        .map(|s| -> Result<Data, Box<dyn std::error::Error>> {
            Ok(match &s.kind {
                SourceKind::File(p) => {
                    let f = File::open(p)
                        .map_err(|e| format!("{}: {}", p.display(), e))?;
                    Data::Mapped(unsafe { Mmap::map(&f) }
                        .map_err(|e| format!("{}: {}", p.display(), e))?)
                }
                SourceKind::Buffered(v) => Data::Owned(v.clone()),
            })
        })
        .collect::<Result<_, _>>()?;

    // Pre-compute per-file entity metadata (boundary segments, label positions).
    let mut entities: Vec<FileEntity> = Vec::new();
    {
        let mut cumulative: u64 = 0;
        for source in &sources {
            let name = match &source.kind {
                SourceKind::File(p) => p
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.to_string_lossy().into_owned()),
                SourceKind::Buffered(_) => "stdin".to_string(),
            };
            let rects = file_rects(
                cumulative,
                cumulative + source.byte_size,
                total_pixels,
                square_pixels,
                num_squares,
                height,
                kh as u8,
            );
            let (pixel_x, pixel_y) = rects_centroid(&rects).unwrap_or_else(|| {
                let mid = cumulative + source.byte_size / 2;
                let sq = mid / square_pixels;
                let (lx, ly): (u32, u32) = h2xy(mid % square_pixels, kh as u8);
                (sq as u32 * height + lx, ly)
            });
            let hue = name_hue(&name);
            let segments = outer_segments(&rects);
            entities.push(FileEntity { name, pixel_x, pixel_y, hue, segments });
            cumulative += source.byte_size;
        }
    }

    // Create the leaf-level tiles directory now (par_iter will create subdirs).
    std::fs::create_dir_all(tile_dir.join(format!("tiles/{max_zoom}")))?;

    let total_tiles = width_tiles as u64 * height_tiles as u64;
    let pb: Option<Arc<ProgressBar>> = if std::io::stderr().is_terminal() {
        let pb = ProgressBar::new(total_tiles);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} tiles ({eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        Some(Arc::new(pb))
    } else {
        None
    };

    // Render all leaf tiles in parallel.  Each tile owns a non-overlapping
    // 256×256-pixel region, so threads never contend on data.
    let errs: Vec<String> = (0..total_tiles)
        .into_par_iter()
        .filter_map(|i| {
            let tx = (i % width_tiles as u64) as u32;
            let ty = (i / width_tiles as u64) as u32;
            let path = tile_dir.join(format!("tiles/{max_zoom}/{tx}/{ty}.png"));
            let result = render_leaf_tile(
                &path,
                tx,
                ty,
                kh as u8,
                height,
                height_tiles,
                square_pixels,
                total,
                &data,
                &cumulative_offsets,
                &pixel_lut,
            );
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
            result.err()
        })
        .collect();

    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }
    if let Some(e) = errs.into_iter().next() {
        return Err(e.into());
    }

    eprintln!("Building pyramid …");
    build_pyramid(&tile_dir.join("tiles"), tile_size, max_zoom, width_tiles, height_tiles)?;

    write_leaflet_html(&tile_dir, world_w, max_zoom, height, &entities)?;

    eprintln!("Tiled output written to {}", tile_dir.display());
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {

    let mut files = args.files;
    if let Some(list_path) = args.file_list {
        let reader: Box<dyn Read> = if list_path == PathBuf::from("-") {
            Box::new(io::stdin())
        } else {
            Box::new(
                File::open(&list_path)
                    .map_err(|e| format!("{}: {}", list_path.display(), e))?,
            )
        };
        for line in BufReader::new(reader).lines() {
            let line = line?;
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                files.push(PathBuf::from(trimmed));
            }
        }
    }

    let num_files = files.len().max(1);
    let (sources, total) = prepare_sources(&files)?;

    if let Some(tile_dir) = args.tiles {
        return run_tiles(sources, total, tile_dir);
    }

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

    // Precomputed 256-entry color lookup table — avoids f32 arithmetic per byte.
    let pixel_lut: [Rgb<u8>; 256] = {
        let mut lut = [Rgb([0u8, 0, 0]); 256];
        for v in 0u16..=255 {
            lut[v as usize] = byte_to_pixel(v as u8);
        }
        lut
    };

    // Work directly with ImageBuffer<Rgb<u8>> to avoid DynamicImage dispatch overhead
    // on every pixel write.
    let mut img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = image::ImageBuffer::new(side, side);

    let window = if args.output.is_none() {
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
                    cancelled_c.store(true, Ordering::Relaxed);
                    break;
                }
            }
            cancelled_c.store(true, Ordering::Relaxed);
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

    // --- Mmap sources for random access (enables intra-file parallelism) ---
    let source_data: Vec<Data> = sources
        .iter()
        .map(|s| -> Result<Data, String> {
            Ok(match &s.kind {
                SourceKind::File(p) => {
                    let f = File::open(p)
                        .map_err(|e| format!("{}: {}", p.display(), e))?;
                    Data::Mapped(unsafe { Mmap::map(&f) }.map_err(|e| e.to_string())?)
                }
                SourceKind::Buffered(v) => Data::Owned(v.clone()),
            })
        })
        .collect::<Result<_, String>>()?;

    // --- Split each source into chunks for fine-grained parallelism ---
    // Targeting ~4 MB per chunk so every core gets work even for single-file input.
    let chunk_bytes = (4 * 1024 * 1024u64).max(stride);
    // Each entry: (fi, src_idx, src_global_byte_start, chunk_b_start, chunk_b_end, chunk_pixel_start)
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

    // --- Background display thread for interactive mode ---
    // Reads the image buffer every 100 ms to show in-progress rendering.
    // Torn frames (reads racing concurrent writes) are acceptable: only the
    // final image must be coherent. The alternative — a mutex on every pixel
    // write — would eliminate the parallelism benefit entirely.
    let stop_display = Arc::new(AtomicBool::new(false));
    let display_thread = if let Some(ref w) = window {
        let img_ptr = img.as_ptr() as usize;
        let stop = Arc::clone(&stop_display);
        let cancelled_disp = Arc::clone(&cancelled);
        let w_c = w.clone();
        let side_c = side;
        Some(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) && !cancelled_disp.load(Ordering::Relaxed) {
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

    // --- Parallel rendering ---
    // Each chunk writes exclusively to a non-overlapping pixel range, so
    // concurrent writes to `img` and `pixel_file` are race-free.
    let img_base = img.as_mut_ptr() as usize;
    let pf_base  = pixel_file.as_mut_ptr() as usize;
    let pb_shared: Option<Arc<ProgressBar>> = pb.map(Arc::new);
    let canvas_u = canvas_size as u64;
    let cancelled_proc = Arc::clone(&cancelled);

    let chunk_results: Vec<(usize, Option<(u32, u32, u32, u32)>)> = chunks
        .par_iter()
        .map(|&(fi, src_idx, src_global_start, chunk_b_start, chunk_b_end, chunk_pixel_start)| -> Result<_, String> {
            if chunk_pixel_start >= canvas_u || cancelled_proc.load(Ordering::Relaxed) {
                return Ok((fi, None));
            }
            let data = &source_data[src_idx];
            let local_start = (chunk_b_start - src_global_start) as usize;
            let local_end   = (chunk_b_end   - src_global_start) as usize;
            let bytes = &data[local_start..local_end];

            let mut cur_byte = chunk_b_start;
            let mut cur_pixel = chunk_pixel_start as usize;
            let mut bbox: Option<(u32, u32, u32, u32)> = None;

            for &b in bytes {
                if cur_byte % stride == 0 {
                    let (x, y): (u32, u32) = h2xy(cur_pixel as u64, k as u8);
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
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
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
            Ok((fi, bbox))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Merge per-chunk bboxes into per-file bboxes.
    for (fi, bbox) in chunk_results {
        if let Some(b) = bbox {
            bboxes[fi] = Some(match bboxes[fi] {
                None => b,
                Some((x0, y0, x1, y1)) => (x0.min(b.0), y0.min(b.1), x1.max(b.2), y1.max(b.3)),
            });
        }
    }

    // Stop background display thread before mutating img further.
    stop_display.store(true, Ordering::Relaxed);
    if let Some(t) = display_thread {
        let _ = t.join();
    }

    if cancelled.load(Ordering::Relaxed) {
        return Ok(());
    }

    if let Some(ref pb) = pb_shared {
        pb.finish_and_clear();
    }

    // When multiple files are given, mark pixels on the border between files black.
    // A border pixel is any pixel whose 4-neighbor was painted by a different file.
    // Reads pixel_file (shared) and writes img via raw pointer — race-free because
    // each row writes only its own pixels and neighbors are read-only.
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

    if let Some(path) = args.output {
        // Output mode: draw all labels after the border pass.
        if !files.is_empty() {
            for (fi, _) in files.iter().enumerate() {
                if let Some(bbox) = bboxes[fi] {
                    draw_file_label(fi, bbox, &files, &mut img, &pixel_file, &font, scale, side);
                }
            }
        }
        DynamicImage::ImageRgb8(img)
            .save(&path)
            .map_err(|e| format!("{}: {}", path.display(), e))?;
    } else if let Some(w) = window {
        // Interactive mode: draw labels then push the final image.
        if !files.is_empty() {
            for (fi, _) in files.iter().enumerate() {
                if let Some(bbox) = bboxes[fi] {
                    draw_file_label(fi, bbox, &files, &mut img, &pixel_file, &font, scale, side);
                }
            }
        }
        if let Err(e) = w.set_image("image-001", DynamicImage::ImageRgb8(img)) {
            if cancelled.load(Ordering::Relaxed) {
                return Ok(());
            }
            return Err(e.into());
        }
        if let Err(e) = w.wait_until_destroyed() {
            if cancelled.load(Ordering::Relaxed) {
                return Ok(());
            }
            return Err(e.into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.is_some() {
        run(args)
    } else {
        show_image::run_context(move || run(args));
    }
}
