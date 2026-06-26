//! 3D (`--3d`) render path: aggregate the source bytes onto a 3D Hilbert curve
//! inside a cube and emit a self-contained Three.js viewer bundle.
//!
//! This is the 3D analog of [`crate::tiled`]. Where the 2D path lays one pixel
//! per byte and builds a tile pyramid, the 3D path lays bytes along a 3D
//! Hilbert curve and aggregates them into a bounded voxel grid — so render and
//! download cost are governed by the grid resolution, not the (potentially
//! many-GB) input size. The viewer ray-marches the grid with opacity encoding
//! density (so the cube's interior is visible) and offers a point-cloud mode
//! for the exact-position view.
//!
//! Output bundle (written to the `--out` directory, deployed verbatim by
//! `--space`): `index.html`, `volume.bin` (RGBA8 `Data3DTexture` payload),
//! `points.bin` (packed positions+colors), and `meta.json`.

pub mod encode;
pub mod html;
pub mod octree;
pub mod shape;
pub mod voxel;

pub use shape::{
    select_volume_shape, HilbertVolumePlugin, VolumeEntity, VolumeLabel, VolumeShape, VoxelBox,
};
pub use voxel::{VoxelCell, VoxelGridMut, VoxelRegistry, VoxelRenderCtx, VoxelRenderer};

use std::path::{Path, PathBuf};

use anyhow::Context;
use image::Rgb;

use crate::color::{build_diff_signed_lut, build_pixel_lut};
use crate::data::{load_source_data, Source};
use crate::geometry;
use crate::layout::LayoutMode;
use crate::registry::{Branding, Registry};
use encode::{VolumeMeta, VoxelAcc};

/// Bytes read per `fetch_range` window during aggregation.
const CHUNK: u64 = 4 * 1024 * 1024;

/// Size cap for the **wholesale fallback** point cloud (`points.bin`) — the
/// one-shot buffer the viewer loads when it can't stream (no octree, or a
/// `file://` open). Kept small for a bounded download. Exact drill-down past
/// this is the streamed LOD octree's job (see [`octree`]), bounded separately
/// by the `--point-budget` knob.
const POINT_BUDGET: u64 = 1_500_000;

/// Max Hilbert order used to place point-cloud positions — caps the coordinate
/// range at `2^16` per axis (plenty for crisp f32-normalized positions).
const POINT_ORDER_CAP: u32 = 16;

struct BuildResult {
    volume_rgba: Vec<u8>,
    points_buf: Vec<u8>,
    points_count: u64,
    max_count: u64,
    focus_center: [f32; 3],
    focus_radius: f32,
    /// Streamed point-LOD octree (byte floor only); `None` for the structured
    /// path, which keeps the wholesale `points.bin`.
    point_octree: Option<octree::PointOctree>,
}

/// Render the 3D viewer bundle for `sources` into `out_dir`.
///
/// Picks a [`VolumeShape`] from the registry (mirroring 2D layout selection):
/// the `i32::MIN` [`HilbertVolumePlugin`] floor runs the legacy whole-stream
/// byte→Hilbert fill, while a higher-priority structured shape (e.g.
/// modelweightvis's `"arch"`) places per-tensor entities and colors them via a
/// [`VoxelRenderer`] that bakes final RGB into each voxel. The viewer's
/// `color_mode` (`"lut"` for the byte path, `"rgb"` for structured) is recorded
/// in `meta.json`.
#[allow(clippy::too_many_arguments)]
pub async fn render_volume(
    sources: Vec<Source>,
    total: u64,
    out_dir: PathBuf,
    title: &str,
    inputs: &[String],
    diff_mode: bool,
    grid_side: u32,
    point_budget: u64,
    mode: LayoutMode,
    registry: &Registry,
    branding: &Branding,
) -> anyhow::Result<()> {
    let pixel_lut = if diff_mode {
        build_diff_signed_lut()
    } else {
        build_pixel_lut()
    };

    // Pick the volume layout. Byte offsets are per-source for the entity path,
    // but `select_volume_shape` (and a downstream's `build`) still wants the
    // cumulative offsets, exactly like `select_layout`.
    let cumulative_offsets = cumulative_offsets(&sources);
    let shape = select_volume_shape(
        &sources,
        &cumulative_offsets,
        total,
        mode,
        diff_mode,
        grid_side,
        registry,
    )?;
    let is_byte = shape.is_byte_volume();
    let actual_extent = shape.grid_extent();
    let [ex, ey, ez] = actual_extent;
    let color_mode = if is_byte { "lut" } else { "rgb" };
    // Pick manifest for the click-to-pick viewer (empty for the byte floor).
    // Captured before `shape` moves into the blocking closure.
    let manifest = shape.manifest();

    if is_byte {
        log::info!("Aggregating {total} bytes into a {ex}³ voxel grid via 3D Hilbert curve...");
    } else {
        log::info!(
            "Rendering structured `{}` 3D volume into a {ex}×{ey}×{ez} voxel box...",
            shape.id()
        );
    }

    // CPU + per-chunk fetch work on the blocking pool, like the single-image
    // path — keeps the tokio runtime free for the `Http`/`Xet`/`LazyDiff`
    // fetches the workers drive via `block_on`. The voxel registry is a cheap
    // Arc-map clone so the blocking closure owns everything it needs.
    let rt = tokio::runtime::Handle::current();
    let voxel_reg = registry.voxel.clone();
    let built = tokio::task::spawn_blocking(move || {
        if is_byte {
            // Byte volumes are cubes (Hilbert needs equal sides); all three
            // axes match, so the cube side is `ex`.
            aggregate_bytes_hilbert(sources, total, ex, point_budget, pixel_lut, rt)
        } else {
            aggregate_entities(
                sources,
                shape.as_ref(),
                actual_extent,
                point_budget,
                &voxel_reg,
                diff_mode,
                rt,
            )
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("volume aggregation join failure: {e}"))??;

    std::fs::create_dir_all(&out_dir)?;
    std::fs::write(out_dir.join("volume.bin"), &built.volume_rgba)?;
    std::fs::write(out_dir.join("points.bin"), &built.points_buf)?;

    // Streamed point-LOD octree (byte floor): two extra files alongside the
    // wholesale points.bin, which stays as the file://-friendly fallback. The
    // viewer prefers the octree when `meta.point_octree` is present.
    let point_octree_meta = if let Some(oct) = &built.point_octree {
        std::fs::write(out_dir.join("points_octree.bin"), &oct.data)?;
        std::fs::write(
            out_dir.join("points_hierarchy.bin"),
            oct.serialize_hierarchy(),
        )?;
        Some(encode::PointOctreeMeta {
            data_file: "points_octree.bin".to_string(),
            hierarchy_file: "points_hierarchy.bin".to_string(),
            record_size: octree::RECORD_SIZE as u32,
            node_count: oct.records.len() as u64,
            order: oct.order,
            grid_log2: octree::POINT_GRID_LOG2,
            total_points: oct.total_points(),
        })
    } else {
        None
    };

    let meta = VolumeMeta {
        title: title.to_string(),
        brand_name: branding.name.to_string(),
        repo_url: branding.repo_url.to_string(),
        grid_extent: actual_extent,
        points: built.points_count,
        total_bytes: total,
        max_count: built.max_count,
        diff_mode,
        color_mode: color_mode.to_string(),
        inputs: inputs.to_vec(),
        focus_center: built.focus_center,
        focus_radius: built.focus_radius,
        lut: pixel_lut.iter().map(|c| c.0).collect(),
        manifest,
        format_version: 2,
        point_octree: point_octree_meta,
    };
    std::fs::write(out_dir.join("meta.json"), serde_json::to_vec(&meta)?)?;
    std::fs::write(
        out_dir.join("index.html"),
        html::build_volume_html(title, inputs, branding),
    )?;

    let (nodes, oct_pts, dropped) = built
        .point_octree
        .as_ref()
        .map(|o| (o.records.len(), o.total_points(), o.dropped))
        .unwrap_or((0, 0, 0));
    log::info!(
        "3D viewer bundle written to {} ({} fallback points; LOD octree: {} points in {} nodes, {} duplicates dropped)",
        out_dir.display(),
        built.points_count,
        oct_pts,
        nodes,
        dropped
    );
    Ok(())
}

/// Prefix-sum of source byte sizes: `out[i]` is the absolute start offset of
/// `sources[i]` in the concatenated stream. Matches the 2D tile path's
/// `cumulative_offsets`, so a downstream's byte arithmetic carries over.
fn cumulative_offsets(sources: &[Source]) -> Vec<u64> {
    let mut offs = Vec::with_capacity(sources.len());
    let mut acc = 0u64;
    for s in sources {
        offs.push(acc);
        acc += s.byte_size;
    }
    offs
}

/// Rebuild `index.html` for an existing 3D bundle from its `meta.json`,
/// without re-aggregating. Mirrors [`crate::tiled::regen_html`].
pub fn regen_html(dir: &Path, branding: &Branding) -> anyhow::Result<()> {
    let meta_path = dir.join("meta.json");
    let bytes = std::fs::read(&meta_path)
        .with_context(|| format!("reading {} (is this a --3d bundle?)", meta_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)?;
    let title = v
        .get("title")
        .and_then(|t| t.as_str())
        .map(String::from)
        .unwrap_or_else(|| branding.name.to_string());
    let inputs: Vec<String> = v
        .get("inputs")
        .and_then(|i| i.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    std::fs::write(
        dir.join("index.html"),
        html::build_volume_html(&title, &inputs, branding),
    )?;
    log::info!("Regenerated 3D viewer index.html in {}", dir.display());
    Ok(())
}

/// Byte-Hilbert floor: a single streaming pass over the concatenated source
/// bytes that populates the voxel grid and samples the point cloud at once.
/// Runs on the blocking pool. Unchanged from the pre-seam path — byte inputs
/// render identically.
fn aggregate_bytes_hilbert(
    sources: Vec<Source>,
    total: u64,
    grid_side: u32,
    octree_budget: u64,
    pixel_lut: [Rgb<u8>; 256],
    rt: tokio::runtime::Handle,
) -> anyhow::Result<BuildResult> {
    let order = grid_side.trailing_zeros(); // grid_side is a power of two
    let cells: u64 = (grid_side as u64).pow(3);
    let luma = encode::luma_lut(&pixel_lut);

    let mut grid = vec![VoxelAcc::default(); cells as usize];
    let mut max_count: u64 = 0;

    // Point cloud: at least as fine as the grid, capped so positions are crisp.
    let point_order =
        geometry::hilbert3d_order_for_cells(total.max(1)).clamp(order, POINT_ORDER_CAP);
    let cells_pt: u128 = 1u128 << (3 * point_order);
    let side_pt = (1u32 << point_order) as f32;
    // Two independent samplings off the same stream: the wholesale fallback
    // cloud (capped at POINT_BUDGET for a bounded one-shot download) and the
    // streamed LOD octree (denser — up to `octree_budget` — since the viewer
    // fetches only on-screen nodes). They share the point grid (`point_order`),
    // so coordinates register. The octree is the 3D analog of the 2D pyramid.
    let stride = (total / POINT_BUDGET).max(1);
    let oct_stride = (total / octree_budget.max(1)).max(1);
    let mut positions: Vec<f32> = Vec::new();
    let mut colors: Vec<u8> = Vec::new();
    let mut next_point_g: u64 = 0;
    let mut next_oct_g: u64 = 0;
    let mut oct_count: u64 = 0;
    let mut octree_builder = octree::PointOctreeBuilder::new(point_order, octree::POINT_GRID_LOG2);

    // First byte not belonging to cell `c`. Two regimes: when the cube can't
    // hold every byte (`total > cells`) each cell aggregates a contiguous byte
    // run; otherwise each byte is its own cell (a contiguous Hilbert prefix, so
    // a small file reads as one solid blob rather than a scattered dust).
    let cell_end = |c: u64| -> u64 {
        if total <= cells {
            c + 1
        } else {
            (((c as u128 + 1) * total as u128).div_ceil(cells as u128)) as u64
        }
    };

    let side = grid_side as usize;
    let flush = |grid: &mut [VoxelAcc], c: u64, acc: &VoxelAcc, max_count: &mut u64| {
        if acc.count == 0 {
            return;
        }
        let [x, y, z] = geometry::hilbert_d2xyz(c, order);
        let lin = x as usize + y as usize * side + z as usize * side * side;
        grid[lin] = *acc;
        if acc.count as u64 > *max_count {
            *max_count = acc.count as u64;
        }
    };

    let mut cur_cell: u64 = 0;
    let mut ce = if total == 0 { 0 } else { cell_end(0) };
    let mut acc = VoxelAcc::default();
    let mut global_start: u64 = 0;

    for src in &sources {
        let data = load_source_data(src)?;
        let size = src.byte_size;
        let mut local: u64 = 0;
        while local < size {
            let len = (size - local).min(CHUNK) as usize;
            let buf = rt.block_on(data.fetch_range(local, len))?;
            for (i, &b) in buf.iter().enumerate() {
                let g = global_start + local + i as u64;

                // Advance the grid cell, flushing the one we leave behind.
                while g >= ce {
                    flush(&mut grid, cur_cell, &acc, &mut max_count);
                    acc = VoxelAcc::default();
                    cur_cell += 1;
                    ce = cell_end(cur_cell);
                }
                acc.count += 1;
                acc.sum_val += b as u64;
                acc.sum_luma += luma[b as usize] as u64;

                // Feed both samplings at their own strides. Compute the
                // point-grid Hilbert distance + color once when either fires.
                let want_flat = g == next_point_g && (positions.len() as u64 / 3) < POINT_BUDGET;
                let want_oct = g == next_oct_g && oct_count < octree_budget;
                if want_flat || want_oct {
                    let cp = if total as u128 <= cells_pt {
                        g
                    } else {
                        ((g as u128) * cells_pt / total as u128) as u64
                    };
                    let c = pixel_lut[b as usize].0;
                    let a = c[0].max(c[1]).max(c[2]); // 0x00 → transparent
                    if want_flat {
                        let [px, py, pz] = geometry::hilbert_d2xyz(cp, point_order);
                        positions.push(px as f32 / side_pt);
                        positions.push(py as f32 / side_pt);
                        positions.push(pz as f32 / side_pt);
                        colors.extend_from_slice(&[c[0], c[1], c[2], a]);
                        next_point_g += stride;
                    }
                    if want_oct {
                        octree_builder.push(cp, [c[0], c[1], c[2], a]);
                        oct_count += 1;
                        next_oct_g += oct_stride;
                    }
                }
            }
            local += len as u64;
        }
        global_start += size;
    }
    // Flush the trailing in-progress cell.
    flush(&mut grid, cur_cell, &acc, &mut max_count);

    // Occupied bounding box (voxel coords) → world-space center + radius, so the
    // viewer frames the data rather than a mostly-empty cube.
    let (focus_center, focus_radius) = occupied_focus(&grid, [grid_side; 3]);

    let volume_rgba = encode::grid_to_rgba(&grid, max_count);
    let points_count = positions.len() as u64 / 3;
    let points_buf = encode::pack_points(&positions, &colors);
    let octree = octree_builder.finish();
    let point_octree = (!octree.records.is_empty()).then_some(octree);

    Ok(BuildResult {
        volume_rgba,
        points_buf,
        points_count,
        max_count,
        focus_center,
        focus_radius,
        point_octree,
    })
}

/// Structured path: render each entity into its voxel box via the matching
/// [`VoxelRenderer`], which bakes final RGBA8 straight into the grid (no shader
/// LUT). arbvis owns the fetch — it reads the whole `[byte_start, +byte_len)`
/// span per entity and hands the bytes to the renderer, which decodes/samples
/// within. (Peak memory is the largest single entity; a future revision can
/// switch to a fetch-on-demand callback for very large tensors.)
fn aggregate_entities(
    sources: Vec<Source>,
    shape: &dyn VolumeShape,
    extent: [u32; 3],
    octree_budget: u64,
    voxel_reg: &VoxelRegistry,
    diff_mode: bool,
    rt: tokio::runtime::Handle,
) -> anyhow::Result<BuildResult> {
    let [ex, ey, ez] = extent;
    let cells = ex as usize * ey as usize * ez as usize;
    let mut grid = vec![VoxelCell::default(); cells];
    let entities = shape.entities().unwrap_or_default();
    let default_renderer_id = shape.id();

    let resolve = |ent: &VolumeEntity| -> anyhow::Result<std::sync::Arc<dyn VoxelRenderer>> {
        voxel_reg
            .renderer(ent.renderer_id)
            .or_else(|| voxel_reg.renderer(default_renderer_id))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no voxel renderer registered for id `{}` (shape `{}`)",
                    ent.renderer_id,
                    default_renderer_id
                )
            })
    };

    // Pass 1: total point-octree weight, to split the budget across entities.
    // `point_weight` must not read bytes (we pass an empty span).
    let mut total_w: u128 = 0;
    for ent in &entities {
        let r = resolve(ent)?;
        let ctx = VoxelRenderCtx {
            entity: ent,
            bytes: &[],
            extent,
            diff_mode,
        };
        total_w += r.point_weight(&ctx) as u128;
    }

    // Pass 2: fetch + bake the dense grid, and (if any renderer opts in) collect
    // per-element points for the streamed LOD octree, mapped into the global box.
    let mut oct_points: Vec<([f32; 3], [u8; 4])> = Vec::new();
    for ent in &entities {
        let renderer = resolve(ent)?;
        let src = sources.get(ent.source_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "entity source_idx {} out of range ({} sources)",
                ent.source_idx,
                sources.len()
            )
        })?;
        let data = load_source_data(src)?;

        // Fetch the entity's byte span (chunked, on the blocking pool).
        let mut bytes = vec![0u8; ent.byte_len as usize];
        let mut off = 0u64;
        while off < ent.byte_len {
            let len = (ent.byte_len - off).min(CHUNK) as usize;
            let chunk = rt.block_on(data.fetch_range(ent.byte_start + off, len))?;
            bytes[off as usize..off as usize + len].copy_from_slice(&chunk);
            off += len as u64;
        }

        let ctx = VoxelRenderCtx {
            entity: ent,
            bytes: &bytes,
            extent,
            diff_mode,
        };
        {
            let mut view = VoxelGridMut::new(&mut grid, extent);
            renderer.render(&ctx, &mut view);
        }

        if total_w > 0 {
            let w = renderer.point_weight(&ctx) as u128;
            let budget = if w > 0 {
                (((octree_budget as u128) * w / total_w) as u64).max(1)
            } else {
                0
            };
            if budget > 0 {
                let bb = ent.bbox;
                let (bw, bh, bz) = (
                    (bb.x1 - bb.x0) as f32,
                    (bb.y1 - bb.y0) as f32,
                    (bb.z1 - bb.z0) as f32,
                );
                // bbox-local [0,1]³ → global normalized [0,1]³ (per axis).
                let mut emit = |local: [f32; 3], rgba: [u8; 4]| {
                    oct_points.push((
                        [
                            (bb.x0 as f32 + local[0] * bw) / ex as f32,
                            (bb.y0 as f32 + local[1] * bh) / ey as f32,
                            (bb.z0 as f32 + local[2] * bz) / ez as f32,
                        ],
                        rgba,
                    ));
                };
                renderer.render_points(&ctx, budget, &mut emit);
            }
        }
    }

    let (focus_center, focus_radius) = shape
        .focus()
        .unwrap_or_else(|| occupied_focus_cells(&grid, extent));

    // Wholesale fallback cloud (one point per occupied voxel) — kept for the
    // `file://` / no-octree path, exactly like the byte floor keeps points.bin.
    let (points_buf, points_count) = points_from_grid(&grid, extent);
    let volume_rgba = encode::pack_voxel_cells(&grid);

    // Streamed LOD octree from the renderer-emitted points. They aren't a
    // Hilbert linearization (the layout places entities freely), so this takes
    // the spatial-sort build path. The cube order resolves both the point count
    // and the box's longest axis.
    let point_octree = if !oct_points.is_empty() {
        let max_ext = ex.max(ey).max(ez).max(2);
        let order_ext = 32 - (max_ext - 1).leading_zeros(); // ceil(log2(max_ext))
        let order = geometry::hilbert3d_order_for_cells(oct_points.len() as u64)
            .max(order_ext)
            .clamp(1, POINT_ORDER_CAP);
        let oct = octree::build_from_normalized_points(&oct_points, order, octree::POINT_GRID_LOG2);
        (!oct.records.is_empty()).then_some(oct)
    } else {
        None
    };

    Ok(BuildResult {
        volume_rgba,
        points_buf,
        points_count,
        max_count: 0,
        focus_center,
        focus_radius,
        point_octree,
    })
}

/// Sample one point per occupied (`a > 0`) voxel — strided down to
/// [`POINT_BUDGET`] — at the voxel's box-cell center, carrying its baked RGBA.
/// Positions are normalized per axis into `[0, 1]`; the viewer scales them by
/// the box's world size. Keeps the structured Points view aligned with the
/// volume.
fn points_from_grid(grid: &[VoxelCell], extent: [u32; 3]) -> (Vec<u8>, u64) {
    let [ex, ey, _ez] = extent;
    let (ex, ey) = (ex as usize, ey as usize);
    let inv = [extent[0] as f32, extent[1] as f32, extent[2] as f32];
    let occupied: Vec<usize> = (0..grid.len()).filter(|&i| grid[i].a > 0).collect();
    let stride = (occupied.len() as u64 / POINT_BUDGET).max(1) as usize;
    let mut positions: Vec<f32> = Vec::new();
    let mut colors: Vec<u8> = Vec::new();
    for &i in occupied.iter().step_by(stride) {
        let (x, y, z) = (i % ex, (i / ex) % ey, i / (ex * ey));
        positions.push((x as f32 + 0.5) / inv[0]);
        positions.push((y as f32 + 0.5) / inv[1]);
        positions.push((z as f32 + 0.5) / inv[2]);
        let c = grid[i];
        colors.extend_from_slice(&[c.r, c.g, c.b, c.a]);
    }
    let count = positions.len() as u64 / 3;
    (encode::pack_points(&positions, &colors), count)
}

/// World-space framing center + radius from an occupied bounding box (per-axis
/// `bmin`/`bmax`, centroid `sum/n`) inside a grid of `extent` voxels. Targets
/// the occupancy **centroid** (mass center — the Hilbert prefix clusters
/// asymmetrically within its bounding box) and sizes the radius so the full
/// occupied bounding box stays in view from that center.
///
/// Voxel `v` on axis `a` maps to world position `((v + 0.5)/extent[a] - 0.5) *
/// scale[a]`, where `scale[a] = extent[a]/max(extent)` is the box's world size
/// on that axis (the viewer scales its longest axis to the unit cube and keeps
/// voxels cubic — matching the shader's `uvw = p/uSize + 0.5`). For a cube all
/// scales are 1, so this reduces to the old `(v + 0.5)/side - 0.5`.
fn box_focus(
    bmin: [u32; 3],
    bmax: [u32; 3],
    sum: [f64; 3],
    n: u64,
    extent: [u32; 3],
) -> ([f32; 3], f32) {
    if n == 0 {
        return ([0.0, 0.0, 0.0], 0.5);
    }
    let maxext = extent[0].max(extent[1]).max(extent[2]) as f32;
    let to_world = |v: f32, axis: usize| {
        let e = extent[axis] as f32;
        ((v + 0.5) / e - 0.5) * (e / maxext)
    };
    let mut center = [0f32; 3];
    let mut radius = 0f32;
    for a in 0..3 {
        let c = to_world((sum[a] / n as f64) as f32, a);
        let lo = to_world(bmin[a] as f32, a);
        let hi = to_world(bmax[a] as f32, a);
        center[a] = c;
        radius = radius.max((c - lo).max(hi - c));
    }
    (center, radius.max(0.02))
}

/// Decode the x-fastest linear index `i` into voxel coordinates for `extent`.
fn voxel_coord(i: usize, extent: [u32; 3]) -> [u32; 3] {
    let (ex, ey) = (extent[0] as usize, extent[1] as usize);
    [
        (i % ex) as u32,
        ((i / ex) % ey) as u32,
        (i / (ex * ey)) as u32,
    ]
}

/// World-space framing for a structured (baked-RGBA) grid — occupancy is `a > 0`.
/// Mirrors [`occupied_focus`] but reads [`VoxelCell`] instead of [`VoxelAcc`].
fn occupied_focus_cells(grid: &[VoxelCell], extent: [u32; 3]) -> ([f32; 3], f32) {
    let mut bmin = [u32::MAX; 3];
    let mut bmax = [0u32; 3];
    let mut sum = [0f64; 3];
    let mut n: u64 = 0;
    for (i, cell) in grid.iter().enumerate() {
        if cell.a == 0 {
            continue;
        }
        n += 1;
        let coord = voxel_coord(i, extent);
        for a in 0..3 {
            bmin[a] = bmin[a].min(coord[a]);
            bmax[a] = bmax[a].max(coord[a]);
            sum[a] += coord[a] as f64;
        }
    }
    box_focus(bmin, bmax, sum, n, extent)
}

/// World-space framing center + radius for the occupied voxels of the byte grid.
/// Falls back to the whole box when nothing is occupied. See [`box_focus`] for
/// the voxel→world mapping.
fn occupied_focus(grid: &[VoxelAcc], extent: [u32; 3]) -> ([f32; 3], f32) {
    let mut bmin = [u32::MAX; 3];
    let mut bmax = [0u32; 3];
    let mut sum = [0f64; 3];
    let mut n: u64 = 0;
    for (i, acc) in grid.iter().enumerate() {
        if acc.count == 0 {
            continue;
        }
        n += 1;
        let coord = voxel_coord(i, extent);
        for a in 0..3 {
            bmin[a] = bmin[a].min(coord[a]);
            bmax[a] = bmax[a].max(coord[a]);
            sum[a] += coord[a] as f64;
        }
    }
    box_focus(bmin, bmax, sum, n, extent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SourceKind;
    use std::sync::Arc;

    fn buffered(bytes: Vec<u8>) -> Source {
        let len = bytes.len() as u64;
        Source {
            file_idx: 0,
            kind: SourceKind::Buffered(bytes),
            byte_size: len,
            name_override: None,
            xet_terms: None,
            extensions: Default::default(),
        }
    }

    /// A half-zero / half-0xFF buffer must produce both fully-transparent
    /// (activity 0) and fully-active (activity 255) occupied voxels, and a
    /// well-formed bundle (volume.bin sized to the grid, all bytes sampled as
    /// points since the input is far under the point budget).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn aggregates_zero_and_ff_split() {
        let mut bytes = vec![0u8; 4096];
        bytes.extend(std::iter::repeat_n(0xFFu8, 4096));
        let total = bytes.len() as u64;
        let dir = tempfile::tempdir().unwrap();
        let grid_side = 32u32;
        render_volume(
            vec![buffered(bytes)],
            total,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            grid_side,
            8_000_000,
            LayoutMode::Auto,
            &Registry::with_defaults(),
            &Branding::default(),
        )
        .await
        .unwrap();

        let vol = std::fs::read(dir.path().join("volume.bin")).unwrap();
        assert_eq!(vol.len() as u64, (grid_side as u64).pow(3) * 4);

        let (mut zero_act, mut full_act, mut occupied) = (false, false, 0u64);
        for px in vol.chunks_exact(4) {
            if px[3] > 0 {
                occupied += 1;
                if px[1] == 0 {
                    zero_act = true; // the 0x00 half → transparent
                }
                if px[1] >= 250 {
                    full_act = true; // the 0xFF half → opaque
                }
            }
        }
        assert_eq!(occupied, total, "one byte per voxel below grid capacity");
        assert!(zero_act && full_act, "expected an activity split");

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(
            meta["grid_extent"],
            serde_json::json!([grid_side, grid_side, grid_side]),
            "byte floor is a cube"
        );
        assert_eq!(meta["points"], total);
        assert_eq!(
            meta["color_mode"], "lut",
            "byte floor must stay LUT-colored"
        );
        assert!(dir.path().join("points.bin").exists());
        assert!(dir.path().join("index.html").exists());
    }

    /// The byte floor must also emit a well-formed streamed point-LOD octree:
    /// `meta.point_octree` present, the two files sized to match the hierarchy
    /// records, and (with enough points to overflow the root) a multi-node tree
    /// whose blocks exactly tile the data file.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn emits_streamed_point_octree() {
        // 200 KB ≫ the root capacity (32³ = 32768) ⇒ the octree must subdivide.
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i * 31 + 7) as u8).collect();
        let total = bytes.len() as u64;
        let dir = tempfile::tempdir().unwrap();
        render_volume(
            vec![buffered(bytes)],
            total,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            8, // small grid ⇒ point grid finer than the voxel grid
            8_000_000,
            LayoutMode::Auto,
            &Registry::with_defaults(),
            &Branding::default(),
        )
        .await
        .unwrap();

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["format_version"], 2, "byte floor ships format v2");
        let po = &meta["point_octree"];
        assert!(po.is_object(), "point_octree descriptor present");
        assert_eq!(po["record_size"], octree::RECORD_SIZE as u64);
        let node_count = po["node_count"].as_u64().unwrap();
        assert!(node_count > 1, "200 KB overflows the root into a tree, got {node_count}");
        assert!(
            po["total_points"].as_u64().unwrap() > 0,
            "octree stores points"
        );

        // Hierarchy file is fixed-size records; data file is the node blocks.
        let hier = std::fs::read(dir.path().join("points_hierarchy.bin")).unwrap();
        let data = std::fs::read(dir.path().join("points_octree.bin")).unwrap();
        assert_eq!(hier.len() as u64, node_count * octree::RECORD_SIZE as u64);

        // Records must tile the data file exactly: every byte covered once.
        let mut covered = 0u64;
        let mut depths = std::collections::BTreeSet::new();
        for chunk in hier.chunks_exact(octree::RECORD_SIZE) {
            let r = octree::NodeRecord::read_le(chunk);
            assert!(
                r.byte_offset + r.byte_length as u64 <= data.len() as u64,
                "record block out of bounds"
            );
            assert_eq!(
                r.byte_length as usize,
                r.point_count as usize * r.stride(),
                "block length matches point count × stride"
            );
            covered += r.byte_length as u64;
            depths.insert(r.depth);
        }
        assert_eq!(covered, data.len() as u64, "node blocks tile the data file");
        assert!(depths.contains(&0), "a root node exists");
        assert!(depths.len() > 1, "multiple LOD levels present");

        // The wholesale fallback is still emitted.
        assert!(dir.path().join("points.bin").exists());
    }

    // A structured VolumeShape whose single entity paints a 4³ box; its
    // VoxelRenderer bakes a fixed RGBA into the cube. Exercises the entity path
    // end to end: shape selection (priority over the floor), per-entity fetch,
    // voxel-renderer dispatch, baked-RGB packing, and the `"rgb"` color_mode +
    // dropped point cloud in `meta.json`.
    struct TestVolume {
        extent: [u32; 3],
    }
    impl VolumeShape for TestVolume {
        fn id(&self) -> &'static str {
            "test-vol"
        }
        fn grid_extent(&self) -> [u32; 3] {
            self.extent
        }
        fn entities(&self) -> Option<Vec<VolumeEntity>> {
            Some(vec![VolumeEntity {
                source_idx: 0,
                byte_start: 0,
                byte_len: 16,
                bbox: VoxelBox {
                    x0: 0,
                    y0: 0,
                    z0: 0,
                    x1: 4,
                    y1: 4,
                    z1: 4,
                },
                renderer_id: "test-vox",
                extra: Box::new(()),
            }])
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct TestVolPlugin;
    impl crate::registry::VolumeShapePlugin for TestVolPlugin {
        fn id(&self) -> &'static str {
            "test-vol"
        }
        fn priority(&self) -> i32 {
            1000
        }
        fn applicable(&self, _ctx: &crate::registry::LayoutBuildCtx<'_>) -> bool {
            true
        }
        fn build(&self, ctx: &crate::registry::LayoutBuildCtx<'_>) -> Option<Box<dyn VolumeShape>> {
            // An anisotropic box derived from the requested resolution, to
            // exercise the non-cube path end to end.
            let s = ctx.grid_side;
            Some(Box::new(TestVolume {
                extent: [s, s * 2, s / 2],
            }))
        }
    }

    struct TestVox;
    impl VoxelRenderer for TestVox {
        fn id(&self) -> &'static str {
            "test-vox"
        }
        fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>) {
            // Prove arbvis fetched the entity span before dispatch.
            assert_eq!(ctx.bytes.len(), 16);
            let bb = ctx.entity.bbox;
            for z in bb.z0..bb.z1 {
                for y in bb.y0..bb.y1 {
                    for x in bb.x0..bb.x1 {
                        grid.put(
                            x,
                            y,
                            z,
                            VoxelCell {
                                r: 10,
                                g: 20,
                                b: 30,
                                a: 200,
                            },
                        );
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structured_entities_bake_rgb() {
        let mut reg = Registry::with_defaults();
        reg.volume_shapes.push(Arc::new(TestVolPlugin));
        reg.voxel.register_renderer(Arc::new(TestVox));

        let dir = tempfile::tempdir().unwrap();
        let grid_side = 8u32;
        render_volume(
            vec![buffered(vec![0u8; 16])],
            16,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            grid_side,
            8_000_000,
            LayoutMode::Auto,
            &reg,
            &Branding::default(),
        )
        .await
        .unwrap();

        // TestVolPlugin builds an anisotropic [s, 2s, s/2] box from grid_side.
        let expect_extent = [grid_side, grid_side * 2, grid_side / 2];
        let expect_cells =
            expect_extent[0] as u64 * expect_extent[1] as u64 * expect_extent[2] as u64;

        let vol = std::fs::read(dir.path().join("volume.bin")).unwrap();
        assert_eq!(vol.len() as u64, expect_cells * 4);
        let mut filled = 0u64;
        for px in vol.chunks_exact(4) {
            if px[3] == 200 {
                assert_eq!(
                    [px[0], px[1], px[2]],
                    [10, 20, 30],
                    "baked RGB survives verbatim"
                );
                filled += 1;
            }
        }
        assert_eq!(filled, 4 * 4 * 4, "the 4³ entity box should be baked");

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        assert_eq!(meta["color_mode"], "rgb");
        assert_eq!(
            meta["points"], 64,
            "one point per occupied voxel (the 4³ baked box)"
        );
        assert_eq!(
            meta["grid_extent"],
            serde_json::json!(expect_extent),
            "structured shape keeps its anisotropic box"
        );
        assert!(
            meta.get("point_octree").map(|v| v.is_null()).unwrap_or(true),
            "a renderer without render_points gets the wholesale fallback, no octree"
        );
    }

    /// A structured renderer that opts into the LOD octree via `point_weight` +
    /// `render_points` must produce a streamed point octree in the bundle (the
    /// modelweightvis drill-down path, exercised here without that crate).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structured_render_points_builds_octree() {
        // Reuses TestVolume/TestVolPlugin (one entity, a 4³ box) but binds the
        // "test-vox" id to a renderer that emits points.
        struct PtVox;
        impl VoxelRenderer for PtVox {
            fn id(&self) -> &'static str {
                "test-vox"
            }
            fn render(&self, ctx: &VoxelRenderCtx<'_>, grid: &mut VoxelGridMut<'_>) {
                let bb = ctx.entity.bbox;
                for z in bb.z0..bb.z1 {
                    for y in bb.y0..bb.y1 {
                        for x in bb.x0..bb.x1 {
                            grid.put(x, y, z, VoxelCell { r: 1, g: 2, b: 3, a: 255 });
                        }
                    }
                }
            }
            fn point_weight(&self, _ctx: &VoxelRenderCtx<'_>) -> u64 {
                512 // "element count"
            }
            fn render_points(
                &self,
                _ctx: &VoxelRenderCtx<'_>,
                budget: u64,
                emit: &mut dyn FnMut([f32; 3], [u8; 4]),
            ) {
                // An 8³ lattice of distinct points in the entity's bbox.
                let n = budget.min(512) as usize;
                for k in 0..n {
                    let x = (k % 8) as f32 / 8.0;
                    let y = ((k / 8) % 8) as f32 / 8.0;
                    let z = ((k / 64) % 8) as f32 / 8.0;
                    emit([x, y, z], [200, 100, 50, 255]);
                }
            }
        }

        let mut reg = Registry::with_defaults();
        reg.volume_shapes.push(Arc::new(TestVolPlugin));
        reg.voxel.register_renderer(Arc::new(PtVox));

        let dir = tempfile::tempdir().unwrap();
        render_volume(
            vec![buffered(vec![0u8; 16])],
            16,
            dir.path().to_path_buf(),
            "test",
            &[],
            false,
            8,
            8_000_000,
            LayoutMode::Auto,
            &reg,
            &Branding::default(),
        )
        .await
        .unwrap();

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("meta.json")).unwrap()).unwrap();
        let po = &meta["point_octree"];
        assert!(po.is_object(), "render_points opts into the streamed octree");
        assert!(po["node_count"].as_u64().unwrap() >= 1);
        assert!(po["total_points"].as_u64().unwrap() > 0);
        assert!(dir.path().join("points_octree.bin").exists());
        assert!(dir.path().join("points_hierarchy.bin").exists());
        // The per-voxel fallback cloud is still emitted (file:// path).
        assert!(dir.path().join("points.bin").exists());
    }
}
