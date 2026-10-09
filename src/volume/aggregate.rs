//! Aggregation for the 3D render path: turn `Source` bytes into the dense
//! voxel grid, the optional sparse brick pool, and the per-bundle framing
//! metrics that `render_volume` bakes into `volume.bin` / `bricks.bin` /
//! `meta.json`. Split out of `volume/mod.rs` so the render driver and the
//! byte/structured aggregation units read separately.

use std::io::Write;
use std::path::Path;

use anyhow::Context;
use image::Rgb;

use super::{brick, encode, part_path, BuildResult, COARSE_CAP, SLAB_BUDGET_BYTES};
use crate::data::{load_source_data, Source};
use crate::geometry;
use crate::volume::encode::VoxelAcc;
use crate::volume::shape::{VolumeEntity, VolumeShape};
use crate::volume::voxel::{VoxelCell, VoxelGridMut, VoxelRegistry, VoxelRenderCtx, VoxelRenderer};

/// Seal a streamed `bricks.bin`: flush the builder into the staged
/// `<file>.part`, then rename it into place. On failure the partial file is
/// removed, so a killed or failed run never leaves a truncated `bricks.bin`
/// for the viewer (or a later rerun) to read as if complete.
pub(super) fn seal_streamed_bricks<W: Write>(
    bb: brick::BrickBuilder<W>,
    out_dir: &Path,
) -> anyhow::Result<brick::BrickVolume> {
    let final_path = out_dir.join("bricks.bin");
    let part = part_path(&final_path);
    match bb.finish_streaming() {
        Ok((bv, _writer)) => {
            std::fs::rename(&part, &final_path)
                .with_context(|| format!("sealing {}", final_path.display()))?;
            Ok(bv)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(anyhow::Error::new(e).context(format!("streaming bricks to {}", part.display())))
        }
    }
}

/// Bytes read per `fetch_range` window during aggregation.
const CHUNK: u64 = 4 * 1024 * 1024;

/// Byte-Hilbert floor: a single streaming pass over the concatenated source
/// bytes that populates the voxel grid (and, with `--volume-res`, the
/// higher-resolution sparse brick pool). Runs on the blocking pool.
pub(super) fn aggregate_bytes_hilbert(
    sources: Vec<Source>,
    total: u64,
    grid_side: u32,
    volume_res: u32,
    pixel_lut: [Rgb<u8>; 256],
    rt: tokio::runtime::Handle,
    out_dir: &Path,
) -> anyhow::Result<BuildResult> {
    let order = grid_side.trailing_zeros(); // grid_side is a power of two
    let cells: u64 = (grid_side as u64).pow(3);
    let luma = encode::luma_lut(&pixel_lut);

    let mut grid = vec![VoxelAcc::default(); cells as usize];
    let mut max_count: u64 = 0;

    // Optional higher-resolution sparse brick pool (--volume-res > --grid).
    // Built streaming in Hilbert order — one open brick at a time, O(brick)
    // memory — so the *volume* can exceed the dense grid for sparse data. The
    // dense grid above is still built (coarser) for histograms + pick.
    let order_v = if volume_res > grid_side {
        volume_res.trailing_zeros()
    } else {
        0
    };
    let cells_v: u128 = if order_v > 0 {
        1u128 << (3 * order_v)
    } else {
        0
    };
    // The streamed brick pool writes each finished brick straight to bricks.bin
    // (append-only, O(one brick) RAM) as the Hilbert curve advances.
    // Bulk regime: when `total > cells_v` several bytes share a voxel on
    // average, so the builder can amortize its Hilbert decode over a per-brick
    // LUT and we can advance `cp_v` by threshold instead of a 128-bit mul/div
    // per byte. In the sparse regime (`total <= cells_v`) `cp_v == g` exactly.
    let bulk = cells_v > 0 && total as u128 > cells_v;
    let mut brick_builder = if order_v > 0 {
        let w = std::io::BufWriter::new(std::fs::File::create(part_path(
            &out_dir.join("bricks.bin"),
        ))?);
        Some(brick::BrickBuilder::new(
            order_v,
            brick::BRICK,
            luma,
            w,
            bulk,
        ))
    } else {
        None
    };

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

    // Bulk-regime cursor: `cp_v` is the voxel of byte `g`, i.e.
    // `g·cells_v/total` floored; `next_g` is the first `g` that maps past
    // `cp_v`, so the 128-bit div runs once per voxel step instead of per byte.
    let mut cp_v: u64 = 0;
    let mut next_g: u64 = if bulk {
        (total as u128).div_ceil(cells_v) as u64
    } else {
        0
    };

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

                // Feed the high-resolution sparse brick pool (every byte, mapped
                // to its voxel on the 2^order_v cube).
                if let Some(bb) = brick_builder.as_mut() {
                    if bulk {
                        while g >= next_g {
                            cp_v += 1;
                            next_g =
                                (((cp_v as u128 + 1) * total as u128).div_ceil(cells_v)) as u64;
                        }
                        bb.push(cp_v, b);
                    } else {
                        bb.push(g, b);
                    }
                }
            }
            local += len as u64;
        }
        global_start += size;
    }
    // Flush the trailing in-progress cell.
    flush(&mut grid, cur_cell, &acc, &mut max_count);

    let volume_rgba = encode::grid_to_rgba(&grid, max_count);
    // Finish the streamed pool: flush bricks.bin and assemble the octree. Bricks
    // were already written to disk, so nothing is dropped — a full disk surfaces
    // as an IO error here rather than truncating detail.
    let bricks = match brick_builder {
        Some(bb) => Some(seal_streamed_bricks(bb, out_dir)?),
        None => None,
    };

    // Camera framing. Stream the *fine* focus (the octree builder's occupied
    // bbox): for a small file at high resolution the coarse grid fills the whole
    // cube while the fine data is a tiny Hilbert-prefix corner, so framing from
    // the coarse grid would aim the camera at empty space. Non-streamed frames
    // from the dense grid as before.
    let (focus_center, focus_radius) = match &bricks {
        Some(bv) if bv.streamed => (bv.focus_center, bv.focus_radius),
        _ => occupied_focus(&grid, [grid_side; 3]),
    };

    Ok(BuildResult {
        volume_rgba,
        // The dense `volume.bin` is the (capped) coarse cube; the streamed pool
        // carries the fine detail at its own `vol_dim`.
        grid_extent: [grid_side; 3],
        max_count,
        focus_center,
        focus_radius,
        bricks,
    })
}

/// Structured path: render each entity into its voxel box via the matching
/// [`VoxelRenderer`], which bakes final RGBA8 straight into the grid (no shader
/// LUT). arbvis owns the fetch — it reads the whole `[byte_start, +byte_len)`
/// span per entity and hands the bytes to the renderer, which decodes/samples
/// within. (Peak memory is the largest single entity; a future revision can
/// switch to a fetch-on-demand callback for very large tensors.)
#[allow(clippy::too_many_arguments)]
pub(super) fn aggregate_entities(
    sources: Vec<Source>,
    shape: &dyn VolumeShape,
    extent: [u32; 3],
    voxel_reg: &VoxelRegistry,
    diff_mode: bool,
    stream: bool,
    rt: tokio::runtime::Handle,
    out_dir: &Path,
) -> anyhow::Result<BuildResult> {
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

    if stream {
        // Dense-grid-free streamed path: process the volume in brick-aligned
        // Z-slabs so peak RAM is one slab (never `extent³`). Per slab, render
        // only the entities whose bbox intersects it into a slab-sized buffer,
        // brick that slab straight to `bricks.bin`, and fold it into the coarse
        // downsample. Frame from the fine occupied region (the aggregator's bbox).
        let [ex, ey, ez] = extent;
        let brick = brick::BRICK;

        // slab_depth: bound the slab buffer to ~SLAB_BUDGET_BYTES, a multiple of
        // BRICK (so no brick row straddles a boundary), clamped to [BRICK, ez↑].
        let layer = ex as usize * ey as usize * std::mem::size_of::<VoxelCell>();
        let raw = (SLAB_BUDGET_BYTES / layer.max(1)) as u32;
        let max_depth = ez.div_ceil(brick) * brick; // ez rounded up to a brick multiple
        let slab_depth = (raw / brick * brick).clamp(brick, max_depth);

        // Each entity's z-interval from its bbox (its authoritative target region,
        // known before dispatch), for scheduling and residency eviction.
        let ivals: Vec<(u32, u32)> = entities.iter().map(|e| (e.bbox.z0, e.bbox.z1)).collect();

        // Fetch-once residency: fetch an entity's bytes on the first slab it
        // intersects and evict once the slab front passes its bbox.z1. Live bytes
        // = Σ spans of entities overlapping the current slab.
        let mut cache: std::collections::HashMap<usize, std::sync::Arc<Vec<u8>>> =
            std::collections::HashMap::new();

        let part = part_path(&out_dir.join("bricks.bin"));
        let built = (|| -> anyhow::Result<BuildResult> {
            let mut agg = brick::StreamBrickAgg::new(extent, brick);
            let mut w = std::io::BufWriter::new(std::fs::File::create(&part)?);
            let ce = encode::coarse_extent(extent, COARSE_CAP);
            let mut coarse = encode::CoarseAcc::new(extent, ce);

            let mut z0 = 0u32;
            while z0 < ez {
                let z1 = (z0 + slab_depth).min(ez);
                let depth = (z1 - z0) as usize;
                let mut slab = vec![VoxelCell::default(); ex as usize * ey as usize * depth];

                for (i, ent) in entities.iter().enumerate() {
                    let (ez0, ez1) = ivals[i];
                    if ez0 >= z1 || ez1 <= z0 {
                        continue; // bbox doesn't intersect this slab
                    }
                    let renderer = resolve(ent)?;
                    let bytes = match cache.get(&i) {
                        Some(b) => b.clone(),
                        None => {
                            let b = std::sync::Arc::new(fetch_entity_bytes(&sources, ent, &rt)?);
                            cache.insert(i, b.clone());
                            b
                        }
                    };
                    let ctx = VoxelRenderCtx {
                        entity: ent,
                        bytes: &bytes[..],
                        extent,
                        diff_mode,
                    };
                    let mut view = VoxelGridMut::slab(&mut slab, extent, z0, z1);
                    renderer.render_window(&ctx, &mut view, z0..z1);
                }

                let slab_rgba = encode::pack_voxel_cells(&slab);
                agg.add_slab(&slab_rgba, z0, z1, &mut w)?;
                coarse.add_slab(&slab_rgba, z0, z1);

                cache.retain(|&i, _| ivals[i].1 > z1); // evict entities behind the front
                z0 = z1; // free `slab`/`slab_rgba`
            }
            w.flush()?;
            let bricks = agg.finish();
            Ok(BuildResult {
                volume_rgba: coarse.finish(),
                grid_extent: ce,
                max_count: 0,
                focus_center: bricks.focus_center,
                focus_radius: bricks.focus_radius,
                bricks: Some(bricks),
            })
        })();
        match built {
            Ok(res) => {
                let final_path = out_dir.join("bricks.bin");
                std::fs::rename(&part, &final_path)
                    .with_context(|| format!("sealing {}", final_path.display()))?;
                return Ok(res);
            }
            Err(e) => {
                // Drop the staged partial so no truncated bricks.bin is left
                // for the viewer or a rerun to read as if complete.
                let _ = std::fs::remove_file(&part);
                return Err(e);
            }
        }
    }

    // Below the coarse cap: simple dense (non-streamed) path — render every
    // entity into one full grid (bounded by `extent ≤ COARSE_CAP`), then derive
    // bricks from it in render_volume.
    let cells = extent[0] as usize * extent[1] as usize * extent[2] as usize;
    let mut grid = vec![VoxelCell::default(); cells];
    for ent in &entities {
        let renderer = resolve(ent)?;
        let bytes = fetch_entity_bytes(&sources, ent, &rt)?;
        let ctx = VoxelRenderCtx {
            entity: ent,
            bytes: &bytes,
            extent,
            diff_mode,
        };
        let mut view = VoxelGridMut::new(&mut grid, extent);
        renderer.render(&ctx, &mut view);
    }

    let full_rgba = encode::pack_voxel_cells(&grid);
    let (focus_center, focus_radius) = shape
        .focus()
        .unwrap_or_else(|| occupied_focus_cells(&grid, extent));

    Ok(BuildResult {
        volume_rgba: full_rgba,
        grid_extent: extent,
        max_count: 0,
        focus_center,
        focus_radius,
        bricks: None,
    })
}

/// Fetch an entity's `[byte_start, +byte_len)` span (chunked, on the blocking
/// pool via `rt.block_on`). Peak transient RAM is the entity's own span.
pub(super) fn fetch_entity_bytes(
    sources: &[Source],
    ent: &VolumeEntity,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Vec<u8>> {
    let src = sources.get(ent.source_idx).ok_or_else(|| {
        anyhow::anyhow!(
            "entity source_idx {} out of range ({} sources)",
            ent.source_idx,
            sources.len()
        )
    })?;
    let data = load_source_data(src)?;
    let mut bytes = vec![0u8; ent.byte_len as usize];
    let mut off = 0u64;
    while off < ent.byte_len {
        let len = (ent.byte_len - off).min(CHUNK) as usize;
        let chunk = rt.block_on(data.fetch_range(ent.byte_start + off, len))?;
        bytes[off as usize..off as usize + len].copy_from_slice(&chunk);
        off += len as u64;
    }
    Ok(bytes)
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
pub(crate) fn box_focus(
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

    #[test]
    fn voxel_coord_is_x_fastest() {
        let extent = [4u32, 2, 2];
        assert_eq!(voxel_coord(0, extent), [0, 0, 0]);
        assert_eq!(voxel_coord(3, extent), [3, 0, 0]);
        assert_eq!(voxel_coord(4, extent), [0, 1, 0]);
        assert_eq!(voxel_coord(7, extent), [3, 1, 0]);
        assert_eq!(voxel_coord(8, extent), [0, 0, 1]);
        assert_eq!(voxel_coord(15, extent), [3, 1, 1]);
    }

    #[test]
    fn box_focus_empty_grid_falls_back_to_unit_center() {
        let (center, radius) = box_focus([u32::MAX; 3], [0; 3], [0.0; 3], 0, [8, 8, 8]);
        assert_eq!(center, [0.0, 0.0, 0.0]);
        assert_eq!(radius, 0.5);
    }

    #[test]
    fn box_focus_single_occupied_voxel_gets_minimum_radius() {
        // Single occupied voxel (1,1,1): centroid == the voxel, lo == hi, so
        // the raw radius is 0 and the 0.02 floor applies.
        let (center, radius) = box_focus([1, 1, 1], [1, 1, 1], [1.0, 1.0, 1.0], 1, [2, 2, 2]);
        assert_eq!(center, [0.25, 0.25, 0.25]);
        assert!((radius - 0.02).abs() < 1e-6);
    }

    #[test]
    fn box_focus_cube_maps_voxels_symmetrically() {
        // Occupied corners (0,0,0) and (1,1,1) of a 2³ grid: the centroid maps
        // to the box center and the radius reaches each corner (0.25).
        let (center, radius) = box_focus([0, 0, 0], [1, 1, 1], [1.0, 1.0, 1.0], 2, [2, 2, 2]);
        assert!(center.iter().all(|c| c.abs() < 1e-6));
        assert!((radius - 0.25).abs() < 1e-6);
    }

    #[test]
    fn box_focus_non_cube_extent_scales_axes_by_maxext() {
        // extent [4,2,2]: the x axis spans the full unit width while y and z
        // are compressed by 2/4, matching the viewer's unit-cube normalization.
        let (center, radius) = box_focus([0, 0, 0], [3, 1, 1], [6.0, 1.0, 1.0], 2, [4, 2, 2]);
        // Centroid x = 6/2 = 3 → (3.5/4 - 0.5) = 0.375.
        assert!((center[0] - 0.375).abs() < 1e-6);
        // Centroid y = z = 0.5 → ((1.0/2 - 0.5) * 0.5) = 0.0.
        assert!(center[1].abs() < 1e-6 && center[2].abs() < 1e-6);
        // Radius spans centroid→bbox on x: c=0.375, lo=-0.375 → 0.75.
        assert!((radius - 0.75).abs() < 1e-6);
    }

    #[test]
    fn box_focus_asymmetric_cluster_targets_centroid_not_bbox_center() {
        // Occupied voxels at x = 0..3 of a 4³ grid (y=z=0): the centroid sits
        // at x = 1.5 → world (2.0/4 - 0.5) = 0.0 — the bbox center would be
        // x = 1.0 → (1.5/4 - 0.5) = -0.125. Centroid wins.
        let (center, radius) = box_focus([0, 0, 0], [3, 0, 0], [6.0, 0.0, 0.0], 4, [4, 4, 4]);
        assert!(center[0].abs() < 1e-6);
        assert!((center[1] - (-0.375)).abs() < 1e-6);
        // Radius covers from centroid to the bbox far end: 0.375 on x.
        assert!((radius - 0.375).abs() < 1e-6);
    }

    #[test]
    fn occupied_focus_scans_linear_grid_for_occupancy() {
        // 2³ grid; voxels at linear indices 3 (1,1,1) and 6 (0,1,1) are set.
        let mut grid = vec![VoxelAcc::default(); 8];
        grid[3].count = 1;
        grid[6].count = 5;
        let (center, radius) = occupied_focus(&grid, [2, 2, 2]);
        // Occupied voxels: (1,1,0) at index 3 and (0,1,1) at index 6.
        // Centroid = (0.5, 1, 0.5) → world (0, 0.25, 0) on a unit cube.
        assert!(center[0].abs() < 1e-6);
        assert!((center[1] - 0.25).abs() < 1e-6);
        assert!(center[2].abs() < 1e-6);
        // bmin = (0,1,0), bmax = (1,1,1): farthest extent from the centroid
        // is 0.25 (on x and z).
        assert!((radius - 0.25).abs() < 1e-6);
    }

    #[test]
    fn occupied_focus_all_zero_grid_falls_back() {
        let grid = vec![VoxelAcc::default(); 8];
        let (center, radius) = occupied_focus(&grid, [2, 2, 2]);
        assert_eq!(center, [0.0, 0.0, 0.0]);
        assert_eq!(radius, 0.5);
    }

    #[test]
    fn occupied_focus_cells_uses_alpha_as_occupancy() {
        // Same layout as the VoxelAcc test, but a = 0 marks empty cells.
        let mut grid = vec![VoxelCell::default(); 8];
        grid[3].a = 255;
        grid[6].a = 1; // any nonzero alpha counts
        let (center, radius) = occupied_focus_cells(&grid, [2, 2, 2]);
        assert!(center[0].abs() < 1e-6);
        assert!((center[1] - 0.25).abs() < 1e-6);
        assert!(center[2].abs() < 1e-6);
        assert!((radius - 0.25).abs() < 1e-6);
    }

    #[test]
    fn occupied_focus_cells_all_transparent_falls_back() {
        let grid = vec![VoxelCell::default(); 8];
        let (center, radius) = occupied_focus_cells(&grid, [2, 2, 2]);
        assert_eq!(center, [0.0, 0.0, 0.0]);
        assert_eq!(radius, 0.5);
    }
}
