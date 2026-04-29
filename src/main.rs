use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use ab_glyph::{FontRef, PxScale};
use clap::Parser;
use fast_hilbert::h2xy;
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

impl Source {
    fn open(self) -> Result<Box<dyn Read>, Box<dyn std::error::Error>> {
        Ok(match self.kind {
            SourceKind::Buffered(buf) => Box::new(io::Cursor::new(buf)),
            SourceKind::File(ref path) => Box::new(BufReader::new(
                File::open(path)
                    .map_err(|e| format!("{}: {}", path.display(), e))?,
            )),
        })
    }
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
        let size = std::fs::metadata(path)
            .map_err(|e| format!("{}: {}", path.display(), e))?
            .len();
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

struct TileBuffer {
    img: image::ImageBuffer<Rgb<u8>, Vec<u8>>,
    dirty: bool,
}

struct TileWriter {
    dir: PathBuf,
    tile_size: u32,
    buffers: HashMap<(u32, u32, u32), TileBuffer>,
    max_buffered: usize,
}

impl TileWriter {
    fn new(dir: PathBuf, tile_size: u32) -> Self {
        std::fs::create_dir_all(&dir).unwrap();
        Self {
            dir,
            tile_size,
            buffers: HashMap::new(),
            max_buffered: 500,
        }
    }

    fn tile_path(&self, z: u32, x: u32, y: u32) -> PathBuf {
        self.dir.join(format!("{z}/{x}/{y}.png"))
    }

    fn get_or_create(&mut self, z: u32, x: u32, y: u32) -> &mut TileBuffer {
        let key = (z, x, y);
        if !self.buffers.contains_key(&key) {
            let path = self.tile_path(z, x, y);
            let img = if path.exists() {
                image::open(&path)
                    .map(|d| d.to_rgb8())
                    .unwrap_or_else(|_| image::ImageBuffer::new(self.tile_size, self.tile_size))
            } else {
                image::ImageBuffer::new(self.tile_size, self.tile_size)
            };
            self.buffers.insert(key, TileBuffer { img, dirty: false });
        }
        self.buffers.get_mut(&key).unwrap()
    }

    fn write_pixel(
        &mut self,
        z: u32,
        x: u32,
        y: u32,
        px: u32,
        py: u32,
        color: Rgb<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let buf = self.get_or_create(z, x, y);
        buf.img.put_pixel(px, py, color);
        buf.dirty = true;
        if self.buffers.len() > self.max_buffered {
            self.flush_all()?;
        }
        Ok(())
    }

    fn flush_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let dir = self.dir.clone();
        for ((z, x, y), buf) in self.buffers.drain() {
            if buf.dirty {
                let path = dir.join(format!("{z}/{x}/{y}.png"));
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                buf.img.save(&path)
                    .map_err(|e| format!("{}: {}", path.display(), e))?;
            }
        }
        Ok(())
    }
}

fn build_pyramid(
    tiles_dir: &Path,
    tile_size: u32,
    max_zoom: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if max_zoom == 0 {
        return Ok(());
    }
    let half = tile_size / 2;
    for z in (0..max_zoom).rev() {
        let child_z = z + 1;
        let parents = 1usize << z;
        for y in 0..parents {
            for x in 0..parents {
                let mut parent = image::ImageBuffer::new(tile_size, tile_size);
                let mut has_data = false;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let cx = 2 * x + dx;
                        let cy = 2 * y + dy;
                        let child_path = tiles_dir.join(format!("{child_z}/{cx}/{cy}.png"));
                        if !child_path.exists() {
                            continue;
                        }
                        let child = image::open(&child_path)?.to_rgb8();
                        has_data = true;
                        for py in 0..half {
                            for px in 0..half {
                                let mut r = 0u32;
                                let mut g = 0u32;
                                let mut b = 0u32;
                                let mut count = 0u32;
                                for sy in 0..2 {
                                    for sx in 0..2 {
                                        let sx_ = (px * 2 + sx) as u32;
                                        let sy_ = (py * 2 + sy) as u32;
                                        if sx_ < child.width() && sy_ < child.height() {
                                            let p = child.get_pixel(sx_, sy_);
                                            r += p[0] as u32;
                                            g += p[1] as u32;
                                            b += p[2] as u32;
                                            count += 1;
                                        }
                                    }
                                }
                                if count > 0 {
                                    let out_x = (dx as u32) * half + px as u32;
                                    let out_y = (dy as u32) * half + py as u32;
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
                        std::fs::create_dir_all(p)?;
                    }
                    parent.save(&parent_path)
                        .map_err(|e| format!("{}: {}", parent_path.display(), e))?;
                }
            }
        }
    }
    Ok(())
}

fn write_leaflet_html(
    dir: &Path,
    side: u32,
    max_zoom: u32,
) -> Result<(), Box<dyn std::error::Error>> {
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
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{
      crs: L.CRS.Simple,
      minZoom: 0,
      maxZoom: {max_zoom},
    }});
    L.tileLayer('tiles/{{z}}/{{x}}/{{y}}.png', {{
      tileSize: 256,
      bounds: [[0, 0], [{side}, {side}]],
      noWrap: true,
      attribution: 'arbvis'
    }}).addTo(map);
    map.fitBounds([[0, 0], [{side}, {side}]]);
  </script>
</body>
</html>"#,
        side = side,
        max_zoom = max_zoom,
    );
    std::fs::write(dir.join("index.html"), html)?;
    Ok(())
}

fn run_tiles(
    sources: Vec<Source>,
    total: u64,
    tile_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let total_usize = total as usize;
    let mut k = 1u32;
    while (1usize << (2 * k)) < total_usize {
        k += 1;
    }
    let side = 1u32 << k;
    let tile_size = 256u32;
    let max_zoom = if k >= 8 { k - 8 } else { 0 };

    let pixel_lut: [Rgb<u8>; 256] = {
        let mut lut = [Rgb([0u8, 0, 0]); 256];
        for v in 0u16..=255 {
            lut[v as usize] = byte_to_pixel(v as u8);
        }
        lut
    };

    let mut writer = TileWriter::new(tile_dir.join("tiles"), tile_size);

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

    let mut pixel_idx = 0usize;

    'outer: for source in sources {
        let mut reader = source.open()?;
        let mut buf = vec![0u8; 65536];
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if pixel_idx >= (side * side) as usize {
                    break 'outer;
                }
                let (x, y): (u32, u32) = h2xy(pixel_idx as u64, k as u8);
                let tx = x / tile_size;
                let ty = y / tile_size;
                let px = x % tile_size;
                let py = y % tile_size;
                writer.write_pixel(max_zoom, tx, ty, px, py, pixel_lut[b as usize])?;
                pixel_idx += 1;
            }
            if let Some(ref pb) = pb {
                pb.inc(n as u64);
            }
        }
    }

    writer.flush_all()?;
    if let Some(ref pb) = pb {
        pb.finish_and_clear();
    }

    eprintln!("Building pyramid …");
    build_pyramid(&tile_dir.join("tiles"), tile_size, max_zoom)?;

    write_leaflet_html(&tile_dir, side, max_zoom)?;

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

    // --- Pre-compute per-file byte and pixel offsets for parallel dispatch ---
    let mut file_meta: Vec<(u64, u64)> = Vec::with_capacity(sources.len());
    {
        let mut b = 0u64;
        let mut p = 0u64;
        for s in &sources {
            file_meta.push((b, p));
            p += sampled_in_range(b, b + s.byte_size, stride);
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

    // --- Parallel file processing ---
    // Each source writes exclusively to pixel indices in [pixel_start, pixel_end),
    // determined by its non-overlapping byte range. Because every distinct
    // pixel_count value maps to a unique Hilbert coordinate, writes to `img` and
    // `pixel_file` across sources never alias — concurrent writes are race-free.
    let img_base = img.as_mut_ptr() as usize;
    let pf_base  = pixel_file.as_mut_ptr() as usize;
    let pb_shared: Option<Arc<ProgressBar>> = pb.map(Arc::new);
    let canvas_u = canvas_size as u64;

    let cancelled_proc = Arc::clone(&cancelled);
    let par_results: Vec<(usize, Option<(u32, u32, u32, u32)>)> = sources
        .into_par_iter()
        .zip(file_meta.into_par_iter())
        .map(|(source, (byte_start, pixel_start))| -> Result<_, String> {
            let fi = source.file_idx;
            if pixel_start >= canvas_u {
                return Ok((fi, None));
            }
            let mut reader = source.open().map_err(|e| e.to_string())?;
            let mut read_buf = vec![0u8; 65536];
            let mut cur_byte = byte_start;
            let mut cur_pixel = pixel_start as usize;
            let mut bbox: Option<(u32, u32, u32, u32)> = None;

            'read: loop {
                if cancelled_proc.load(Ordering::Relaxed) {
                    break;
                }
                let n = reader.read(&mut read_buf).map_err(|e| e.to_string())?;
                if n == 0 {
                    break;
                }
                for &b in &read_buf[..n] {
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
                            Some((x0, y0, x1, y1)) => {
                                (x0.min(x), y0.min(y), x1.max(x), y1.max(y))
                            }
                        });
                        cur_pixel += 1;
                        if cur_pixel >= canvas_size {
                            break 'read;
                        }
                    }
                    cur_byte += 1;
                }
                if let Some(ref pb) = pb_shared {
                    pb.inc(n as u64);
                }
            }
            Ok((fi, bbox))
        })
        .collect::<Result<Vec<_>, String>>()?;

    for (fi, bbox) in par_results {
        bboxes[fi] = bbox;
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
                        img.put_pixel(x, y, Rgb([0, 0, 0]));
                    }
                }
            }
        }
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
